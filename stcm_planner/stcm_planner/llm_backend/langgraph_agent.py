
import ast
import re
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from typing import Union, Literal, Any
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from stcm_planner.prompts import get_critic_prompt
from langchain.prompts import ChatPromptTemplate
from langgraph.managed.is_last_step import RemainingSteps


def _parse_int_args(arg_str: str) -> list[int]:
    args = []
    for part in arg_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            args.append(int(part))
        except ValueError:
            try:
                args.append(int(float(part)))
            except ValueError:
                continue
    return args


def _commands_from_text(content: str) -> list[list]:
    commands = []
    if not content:
        return commands
    for match in re.finditer(r"go_near\(([^)]*)\)", content):
        args = _parse_int_args(match.group(1))
        if args:
            commands.append(["go_near", [args[0]]])
    for match in re.finditer(r"go_between\(([^)]*)\)", content):
        args = _parse_int_args(match.group(1))
        if len(args) >= 2:
            commands.append(["go_between", [args[0], args[1]]])
    return commands


def _extract_object_list_and_query(messages: list[AnyMessage]) -> tuple[list, str] | tuple[None, None]:
    for msg in reversed(messages):
        if not isinstance(msg, HumanMessage):
            continue
        content = msg.content or ""
        if "Object List:" not in content or "User Input:" not in content:
            continue
        obj_index = content.find("Object List:")
        user_index = content.find("User Input:", obj_index)
        if obj_index == -1 or user_index == -1:
            continue
        obj_str = content[obj_index + len("Object List:"):user_index].strip()
        query = content[user_index + len("User Input:"):].strip()
        obj_str = obj_str.strip().strip('"')
        query = query.strip().strip('"')
        try:
            objects = ast.literal_eval(obj_str)
        except Exception:
            continue
        return objects, query
    return None, None


def _find_query_matches(query: str, objects: list) -> list[tuple[int, int]]:
    matches = []
    q = query.lower()
    for obj in objects:
        if not isinstance(obj, (list, tuple)) or len(obj) < 2:
            continue
        obj_id = obj[0]
        obj_name = str(obj[1]).strip().lower()
        if not obj_name:
            continue
        pos = q.find(obj_name)
        if pos == -1:
            continue
        try:
            obj_id_int = int(obj_id)
        except Exception:
            continue
        matches.append((pos, obj_id_int))
    matches.sort(key=lambda item: item[0])
    deduped = []
    seen = set()
    for pos, obj_id_int in matches:
        if obj_id_int in seen:
            continue
        seen.add(obj_id_int)
        deduped.append((pos, obj_id_int))
    return deduped


def _infer_commands_from_query(query: str, objects: list) -> list[list]:
    matches = _find_query_matches(query, objects)
    if not matches:
        return []
    q = query.lower()
    pick_terms = ("pick", "pick up", "pickup", "grab", "fetch", "take", "get")
    bring_terms = ("bring", "deliver", "return", "place", "put", "drop", "carry")
    wants_pick = any(term in q for term in pick_terms)
    wants_bring = any(term in q for term in bring_terms)
    if wants_pick and wants_bring and len(matches) >= 2:
        source = matches[0][1]
        dest = matches[-1][1]
        if source == dest:
            return [["go_near", [source]]]
        return [["go_near", [source]], ["go_near", [dest]]]
    if wants_pick or "go to" in q or "navigate" in q:
        return [["go_near", [matches[0][1]]]]
    return []


def _infer_command_list(messages: list[AnyMessage], content: str) -> list[list]:
    commands = _commands_from_text(content)
    if commands:
        return commands
    objects, query = _extract_object_list_and_query(messages)
    if not objects or not query:
        return []
    return _infer_commands_from_query(query, objects)


def _should_finish_after_tool(messages: list[AnyMessage]) -> bool:
    if not messages:
        return False
    last_msg = messages[-1]
    if not isinstance(last_msg, ToolMessage):
        return False
    content = (last_msg.content or "").lower()
    return (
        "now say 'done'" in content
        or "robot has executed" in content
        or "object has been picked" in content
    )

def build_graph(tools, llm_with_tools, save_graph_png=False):
    tool_names = {tool.name for tool in tools}

    def tool_condition(
        state: MessagesState
    ) -> Literal["tools",
                "instruct_retry",
                    "__end__"]:
        if isinstance(state, list):
            ai_message = state[-1]
        elif isinstance(state, dict) and (messages := state.get("messages", [])):
            ai_message = messages[-1]
        elif messages := getattr(state, "messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if not isinstance(ai_message, AIMessage):
            print(state)
            raise ValueError(f"Expected AIMessage, got {type(ai_message)}")
        if ai_message.tool_calls and len(ai_message.tool_calls) > 0:
            return "tools"
        elif "done" in ai_message.content.lower():
            return "__end__"
        else:
            return "instruct_retry"

    def assistant(state: MessagesState):
        if _should_finish_after_tool(state["messages"]):
            return {"messages": [AIMessage(content="done")]}

        response = llm_with_tools.invoke(state["messages"])
        parsed = False
        # if no tool call, but has content that resembles a tool call, manually parse it
        if len(response.tool_calls) == 0 and "arguments" in response.content and "name" in response.content:
            try:
                content_eval = eval(response.content)
                if isinstance(content_eval, dict):
                    assert "arguments" in content_eval.keys() and "name" in content_eval.keys()
                    arguments = content_eval["arguments"]
                    tool_name = content_eval["name"]
                    response.tool_calls.append({"name": tool_name, "args": arguments, "id": "0"})
                    parsed = True
                elif isinstance(content_eval, list):
                    # make sure arguments and names keys are in each of the dictionaries
                    for idx, tool_call in enumerate(content_eval):
                        assert isinstance(tool_call, dict)
                        assert "arguments" in tool_call.keys() and "name" in tool_call.keys()
                    for idx, tool_call in enumerate(content_eval):
                        if isinstance(tool_call, dict):
                            tool_call["id"] = str(idx)
                            arguments = tool_call["arguments"]
                            tool_name = tool_call["name"]
                            response.tool_calls.append({"name": tool_name, "args": arguments, "id": str(idx)})
                    parsed = True
                if parsed:
                    # human_reminder = HumanMessage(content="Reminder, you should provide a tool call through tool calling API, you should only output content when you are explicitly asked so.")
                    print("**********")
                    print("Manually parsed tool call! at msg: ", response)
                    print("**********")
            except Exception:
                parsed = False

        if len(response.tool_calls) == 0 and not parsed:
            inferred = _infer_command_list(state["messages"], response.content)
            if inferred:
                if "command_robot" in tool_names:
                    response.tool_calls.append(
                        {"name": "command_robot", "args": {"list_of_commands": inferred}, "id": "0"}
                    )
                elif (
                    "pick_object" in tool_names
                    and len(inferred) == 1
                    and inferred[0][0] == "go_near"
                    and inferred[0][1]
                ):
                    response.tool_calls.append(
                        {"name": "pick_object", "args": {"object_id": inferred[0][1][0]}, "id": "0"}
                    )

        return {"messages": [response]}
    
    def instruct_retry(state: MessagesState):
        return {"messages": [HumanMessage(content="Reminder, you should provide a tool call through tool calling API, you should only output content when you are explicitly asked so, your attempt to tool call has failed.")]}
    
    builder = StateGraph(MessagesState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("instruct_retry", instruct_retry)
    # builder.add_node("instruct_function_call", instruct_function_call)

    # Define edges: these determine the control flow
    builder.add_edge(START, "assistant")
    builder.add_edge("instruct_retry", "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call and has DONE -> tools_condition routes to END
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to assistant
        tool_condition,
    )
    builder.add_edge("tools", "assistant")
    # builder.add_edge("instruct_function_call", "assistant")

    # memory = MemorySaver()
    graph = builder.compile(debug=False)
    if save_graph_png:
        with open("graph.png", "wb") as f:
            f.write(graph.get_graph(xray=True).draw_mermaid_png())
    return graph

class ActorCriticState(MessagesState):
    objects: list
    critic_approval: bool = False
    remaining_steps: RemainingSteps

class CriticStructuredOutput(BaseModel):
    approval: bool = Field(description="weather you approve the actor's reasoning or not")
    feedback: str = Field(description="feedback for the actor, what's wrong, what's right")

def build_actor_critic_graph(tools, actor_with_tools, critic_llm, save_graph_png=False):
    tool_names = {tool.name for tool in tools}

    def tool_condition(
        state: ActorCriticState
    ) -> Literal["tools",
                "instruct_retry",
                    "critic",
                    "__end__"]:
        if isinstance(state, list):
            ai_message = state[-1]
        elif isinstance(state, dict) and (messages := state.get("messages", [])):
            ai_message = messages[-1]
        elif messages := getattr(state, "messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        print("Remaining steps: ", state["remaining_steps"])
        if state["remaining_steps"] <= 3:
            return "__end__"
        if ai_message.tool_calls and len(ai_message.tool_calls) > 0:
            return "tools"
        elif "done" in ai_message.content.lower():
            return "critic"
        else:
            return "instruct_retry"
    
    def critic_condition(
        state: ActorCriticState
    ) -> Literal["assistant",
                    "__end__"]:
        if state["critic_approval"] or state["remaining_steps"] <= 10:
            return "__end__"
        else:
            return "assistant"
    
    def assistant(state: ActorCriticState):
        if _should_finish_after_tool(state["messages"]):
            return {"messages": [AIMessage(content="done")]}

        response = actor_with_tools.invoke(state["messages"])
        # print(state["messages"][-1])
        # print("Actor Output:", response)
        parsed = False
        # if no tool call, but has content that resembles a tool call, manually parse it
        if len(response.tool_calls) == 0 and "arguments" in response.content and "name" in response.content:
            try:
                content_eval = eval(response.content)
                if isinstance(content_eval, dict):
                    assert "arguments" in content_eval.keys() and "name" in content_eval.keys()
                    arguments = content_eval["arguments"]
                    tool_name = content_eval["name"]
                    response.tool_calls.append({"name": tool_name, "args": arguments, "id": "0"})
                    parsed = True
                elif isinstance(content_eval, list):
                    # make sure arguments and names keys are in each of the dictionaries
                    for idx, tool_call in enumerate(content_eval):
                        assert isinstance(tool_call, dict)
                        assert "arguments" in tool_call.keys() and "name" in tool_call.keys()
                    for idx, tool_call in enumerate(content_eval):
                        if isinstance(tool_call, dict):
                            tool_call["id"] = str(idx)
                            arguments = tool_call["arguments"]
                            tool_name = tool_call["name"]
                            response.tool_calls.append({"name": tool_name, "args": arguments, "id": str(idx)})
                    parsed = True
                if parsed:
                    # human_reminder = HumanMessage(content="Reminder, you should provide a tool call through tool calling API, you should only output content when you are explicitly asked so.")
                    print("**********")
                    print("Manually parsed tool call! at msg: ", response)
                    print("**********")
            except Exception:
                parsed = False

        if len(response.tool_calls) == 0 and not parsed:
            inferred = _infer_command_list(state["messages"], response.content)
            if inferred:
                if "command_robot" in tool_names:
                    response.tool_calls.append(
                        {"name": "command_robot", "args": {"list_of_commands": inferred}, "id": "0"}
                    )
                elif (
                    "pick_object" in tool_names
                    and len(inferred) == 1
                    and inferred[0][0] == "go_near"
                    and inferred[0][1]
                ):
                    response.tool_calls.append(
                        {"name": "pick_object", "args": {"object_id": inferred[0][1][0]}, "id": "0"}
                    )

        return {"messages": [response]}
    
    def critic(state: ActorCriticState):
        # format the messages to be passed to the critic
        update = {}
        messages = state["messages"]
        parsed_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                parsed_messages.append("Human: " + msg.content)
            elif isinstance(msg, AIMessage):
                parsed_messages.append("Actor: " + msg.content)
            elif isinstance(msg, ToolMessage):
                parsed_messages.append(f"Tool Returned {msg.content}")
        parsed_output = "\n".join(parsed_messages)
        critic_sys_prompt = get_critic_prompt(objects=state["objects"])
        critic_input = [
            SystemMessage(content=critic_sys_prompt),
            HumanMessage(content="The Actor Reasoning Trace is: \n" + parsed_output)
        ]
        critic_output = critic_llm.invoke(critic_input)
        print("Critic Output: ", critic_output)
        if critic_output.approval:
            update["critic_approval"] = True
        update["messages"] = [HumanMessage(content=critic_output.feedback)]
        
        return update

    
    def instruct_retry(state: ActorCriticState):
        return {"messages": [HumanMessage(content="Reminder, you should provide a tool call, you should only output content when you are explicitly asked so! At the end, you should call `command_robot` tool to best of your knowledge!")]}
    
    builder = StateGraph(ActorCriticState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("instruct_retry", instruct_retry)
    builder.add_node("critic", critic)
    # builder.add_node("instruct_function_call", instruct_function_call)

    # Define edges: these determine the control flow
    builder.add_edge(START, "assistant")
    builder.add_edge("instruct_retry", "assistant")
    builder.add_conditional_edges("assistant",tool_condition)
    builder.add_conditional_edges("critic",critic_condition)
    builder.add_edge("tools", "assistant")
    # builder.add_edge("instruct_function_call", "assistant")

    # memory = MemorySaver()
    graph = builder.compile(debug=False)
    if save_graph_png:
        with open("graph.png", "wb") as f:
            f.write(graph.get_graph(xray=True).draw_mermaid_png())
    return graph
