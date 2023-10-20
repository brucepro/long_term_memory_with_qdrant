"""
Long Term Memory Module using a qdrant database.
"""
import gradio as gr

from extensions.long_term_memory_with_qdrant.long_term_memory import LTM
from extensions.long_term_memory_with_qdrant.utils.chat_parsing import clean_character_message
from modules import chat

# === Internal constants (don't change these without good reason) ===
_MIN_ROWS_TILL_RESPONSE = 5
_LAST_BOT_MESSAGE_INDEX = -3
params = {
    "display_name": "Long Term Memory",
    "is_tab": False,
    "limit": 5,
    "address": "http://localhost:6333",
    "query_output": "vdb search results",
    'verbose': True,
}


def state_modifier(state):
    """
    Modifies the state variable, which is a dictionary containing the input
    values in the UI like sliders and checkboxes.
    """
    state['limit'] = params['limit']
    state['address'] = params['address']
    return state


def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """

    prompt_line = chat.generate_chat_prompt(user_input, state, **kwargs)
    prompts = prompt_line.split("\n")
    if params['verbose']:
        print("****initial prompt****\n")
        for count, prompt_line in enumerate(prompts, 1):
            print(f"({count}/{len(prompts)}):  {prompt_line}")

    collection = state['name2'].strip()
    username = state['name1'].strip()
    verbose = params['verbose']
    limit = params['limit']
    address = params['address']
    ltm = LTM(collection, verbose, limit, address=address)
    kwargs["also_return_rows"] = True
    (bot_prompt, bot_prompt_rows) = chat.generate_chat_prompt(
        user_input,
        state,
        **kwargs,
    )
    # === Clean and add new messages to LTM ===
    # Store the bot's last message.
    # Avoid storing any of the baked-in bot template responses
    use_bot_memories = 0
    if len(bot_prompt_rows) >= _MIN_ROWS_TILL_RESPONSE:
        bot_message = bot_prompt_rows[_LAST_BOT_MESSAGE_INDEX]
        clean_bot_message = clean_character_message(state["name2"], bot_message)

        # Store bot message into database
        if len(clean_bot_message) >= 10:
            bot_long_term_memories1 = ltm.store_and_recall(state["name2"],clean_bot_message)
            use_bot_memories = 1
            print("-----------------Bot Memories------------------------")
            print(bot_long_term_memories1)
            print("------------------End Bot Memories-----------------------")



    long_term_memories = ltm.store_and_recall(username,user_input)
    print("--------------User Line Memories---------------------------")
    print(long_term_memories)
    print("---------------End User Line Memories--------------------------")
    
    if use_bot_memories == 1:
        long_term_memories.extend(bot_long_term_memories1)
    
    state['query_output'] = "\n".join(long_term_memories)
    # insert the formated vdb outputs after the context but before the chat his
    # tory, this placement seems to work fine but could be played with.
    prompts[1:1] = long_term_memories

    if params['verbose']:
        print("****final prompt with injected memories****\n")
        for count, prompt_line in enumerate(prompts, 1):
            print(f"({count}/{len(prompts)}):  {prompt_line}")

    prompts = "\n".join(prompts)
    return prompts


def setup():
    """
    Gets executed only once, when the extension is imported.
    """
    pass


def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.

    To learn about gradio components, check out the docs:
    https://gradio.app/docs/
    """

    # Gradio elements
    with gr.Accordion("Long Term Memory"):
        with gr.Row():
            limit = gr.Slider(
                1, 10,
                step=1,
                value=params['limit'],
                label='Long Term Memory Result Count (Top N scoring results)',
                )
            limit.change(lambda x: params.update({'limit': x}), limit, None)
