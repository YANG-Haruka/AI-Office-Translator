import gradio as gr
import os
from translator.excel_translator import ExcelTranslator
from translator.ppt_translator import PptTranslator
from translator.word_translator import WordTranslator
from translator.pdf_translator import PdfTranslator
from llmWrapper.ollama_wrapper import populate_sum_model
from typing import List, Tuple
# Import language configs
from config.languages_config import LANGUAGE_MAP, LABEL_TRANSLATIONS

QUEUE_COUNTER = 0

# 1) Main file translation function
def translate_file(
    file, model, src_lang, dst_lang, use_online, api_key, max_token=768,
    progress=gr.Progress(track_tqdm=True)
):
    """Translate an uploaded file using the chosen model."""
    if file is None:
        return gr.update(value=None, visible=False), "Please select a file to translate."

    if use_online and not api_key:
        return gr.update(value=None, visible=False), "API key is required for online models."

    def progress_callback(progress_value, desc=None):
        progress(progress_value, desc=desc)

    src_lang_code = LANGUAGE_MAP.get(src_lang, "en")
    dst_lang_code = LANGUAGE_MAP.get(dst_lang, "en")

    file_name, file_extension = os.path.splitext(file.name)
    translator_class = {
        ".docx": WordTranslator,
        ".pptx": PptTranslator,
        ".xlsx": ExcelTranslator,
        ".pdf": PdfTranslator
    }.get(file_extension.lower())

    if not translator_class:
        return (
            gr.update(value=None, visible=False),
            f"Unsupported file type '{file_extension}'."
        )

    try:
        translator = translator_class(
            file.name, model, use_online, api_key,
            src_lang_code, dst_lang_code, max_token=max_token
        )
        progress(0, desc="Initializing translation...")

        translated_file_path, missing_counts = translator.process(
            file_name, file_extension, progress_callback=progress_callback
        )
        progress(1, desc="Done!")

        if missing_counts:
            msg = f"Warning: Missing segments for keys: {sorted(missing_counts)}"
            return gr.update(value=translated_file_path, visible=True), msg

        return gr.update(value=translated_file_path, visible=True), "Translation complete."
    except ValueError as e:
        return gr.update(value=None, visible=False), f"Translation failed: {str(e)}"
    except Exception as e:
        return gr.update(value=None, visible=False), f"Error: {str(e)}"


# 2) Load local and online models
local_models = populate_sum_model() or []
online_models = ["deepseekv3"]

def update_model_list_and_api_input(use_online):
    """Switch model options and show/hide API Key."""
    if use_online:
        return (
            gr.update(choices=online_models, value=online_models[0]),
            gr.update(visible=True)
        )
    else:
        default_local_value = local_models[0] if local_models else None
        return (
            gr.update(choices=local_models, value=default_local_value),
            gr.update(visible=False)
        )


# 3) Parse Accept-Language
def parse_accept_language(accept_language: str) -> List[Tuple[str, float]]:
    """Parse Accept-Language into (language, q) pairs."""
    if not accept_language:
        return []
    
    languages = []
    for item in accept_language.split(','):
        item = item.strip()
        if not item:
            continue
        if ';q=' in item:
            lang, q = item.split(';q=')
            q = float(q)
        else:
            lang = item
            q = 1.0
        languages.append((lang, q))
    
    return sorted(languages, key=lambda x: x[1], reverse=True)


def get_user_lang(request: gr.Request) -> str:
    """Return the top user language code that matches LANGUAGE_MAP."""
    accept_lang = request.headers.get("accept-language", "").lower()
    parsed = parse_accept_language(accept_lang)
    
    if not parsed:
        return "en"
    
    highest_lang, _ = parsed[0]
    highest_lang = highest_lang.lower()

    if highest_lang.startswith("ja"):
        return "ja"
    elif highest_lang.startswith(("zh-tw", "zh-hk", "zh-hant")):
        return "zh-Hant"
    elif highest_lang.startswith(("zh-cn", "zh-hans", "zh")):
        return "zh"
    elif highest_lang.startswith("es"):
        return "es"
    elif highest_lang.startswith("fr"):
        return "fr"
    elif highest_lang.startswith("de"):
        return "de"
    elif highest_lang.startswith("it"):
        return "it"
    elif highest_lang.startswith("pt"):
        return "pt"
    elif highest_lang.startswith("ru"):
        return "ru"
    elif highest_lang.startswith("ko"):
        return "ko"
    elif highest_lang.startswith("en"):
        return "en"

    return "en"


# 4) Apply labels based on user language
def set_labels(session_lang: str):
    """Update UI labels according to the chosen language."""
    labels = LABEL_TRANSLATIONS.get(session_lang, LABEL_TRANSLATIONS["en"])
    return {
        src_lang: gr.update(label=labels["Source Language"]),
        dst_lang: gr.update(label=labels["Target Language"]),
        use_online_model: gr.update(label=labels["Use Online Model"]),
        model_choice: gr.update(label=labels["Models"]),
        api_key_input: gr.update(label=labels["API Key"]),
        max_token: gr.update(label=labels["Max Tokens"]),
        file_input: gr.update(label=labels["Upload File"]),
        output_file: gr.update(label=labels["Download Translated File"]),
        status_message: gr.update(label=labels["Status Message"]),
        translate_button: gr.update(value=labels["Translate"]),  # Button uses 'value'
    }

def init_ui(request: gr.Request):
    """Set user language and update labels on page load."""
    user_lang = get_user_lang(request)
    return [user_lang] + list(set_labels(user_lang).values())

def get_queue_status():
    global QUEUE_COUNTER
    if QUEUE_COUNTER > 1:
        return f"排队中，前面还有 {QUEUE_COUNTER - 1} 个用户"
    elif QUEUE_COUNTER == 1:
        return "正在处理当前任务，请稍等..."
    else:
        return "没有其他用户在排队。"

def update_queue_count(increment=True):
    global QUEUE_COUNTER
    if increment:
        QUEUE_COUNTER += 1
    else:
        QUEUE_COUNTER = max(0, QUEUE_COUNTER - 1)

# Modified translation function to handle button state
def translate_file_with_state(
    file, model, src_lang, dst_lang, use_online, api_key, max_token,
    progress=gr.Progress(track_tqdm=True)
):
    """Wrapper for translate_file that handles button state."""
    try:
        # Actual translation
        file_output, status = translate_file(
            file, model, src_lang, dst_lang, use_online, api_key, max_token, progress
        )
        return file_output, status, gr.update(interactive=True)
    except Exception as e:
        return (
            gr.update(value=None, visible=False),
            f"Error: {str(e)}",
            gr.update(interactive=True)
        )

# Build Gradio interface
with gr.Blocks() as demo:
    session_lang = gr.State("en")

    with gr.Row():
        src_lang = gr.Dropdown(
            [
                "English", "中文", "繁體中文", "日本語", "Español", 
                "Français", "Deutsch", "Italiano", "Português", 
                "Русский", "한국어"
            ],
            label="Source Language",
            value="English"
        )
        dst_lang = gr.Dropdown(
            [
                "English", "中文", "繁體中文", "日本語", "Español", 
                "Français", "Deutsch", "Italiano", "Português", 
                "Русский", "한국어"
            ],
            label="Target Language",
            value="English"
        )

    with gr.Row():
        use_online_model = gr.Checkbox(label="Use Online Model", value=False)

    default_local_value = local_models[0] if local_models else None
    model_choice = gr.Dropdown(
        choices=local_models,
        label="Models",
        value=default_local_value
    )
    api_key_input = gr.Textbox(label="API Key", placeholder="Enter your API key here", visible=False)
    max_token = gr.Number(label="Max Tokens", value=768)
    file_input = gr.File(
        label="Upload Office File (.docx, .pptx, .xlsx, .pdf)",
        file_types=[".docx", ".pptx", ".xlsx", ".pdf"]
    )
    output_file = gr.File(label="Download Translated File", visible=False)
    status_message = gr.Textbox(label="Status Message", interactive=False, visible=True)
    queue_status_display = gr.Textbox(label="Queue Status", value="没有其他用户在排队。", interactive=False)
    translate_button = gr.Button("Translate")

    use_online_model.change(
        update_model_list_and_api_input,
        inputs=use_online_model,
        outputs=[model_choice, api_key_input]
    )

    # Handle button state and translation
    translate_button.click(
        lambda: (gr.update(visible=False), None, gr.update(interactive=False)),
        inputs=[],
        outputs=[output_file, status_message, translate_button]
    ).then(
        lambda: (update_queue_count(increment=True)),
        inputs=[],
        outputs=[]
    ).then(
        get_queue_status,
        inputs=[],
        outputs=[queue_status_display]  # 更新队列状态到文本框
    ).then(
        lambda: gr.update(interactive=False),  # 灰色禁用按钮
        inputs=[],
        outputs=[translate_button]
    ).then(
        translate_file_with_state,
        inputs=[
            file_input, model_choice, src_lang, dst_lang,
            use_online_model, api_key_input, max_token
        ],
        outputs=[output_file, status_message, translate_button]
    ).then(
        lambda: update_queue_count(increment=False),
        inputs=[],
        outputs=[]
    ).then(
        get_queue_status,
        inputs=[],
        outputs=[queue_status_display]  # 任务结束后更新状态
    ).then(
        lambda: gr.update(interactive=True),  # 恢复按钮
        inputs=[],
        outputs=[translate_button]
    )

    # On page load, set user language and labels
    demo.load(
        fn=init_ui,
        inputs=None,
        outputs=[
            session_lang, src_lang, dst_lang, use_online_model, 
            model_choice, api_key_input, max_token, file_input, 
            output_file, status_message, translate_button
        ]
    )

# Enable queue with concurrency limit
demo.queue(default_concurrency_limit=1)
demo.launch(server_port=9980, share=False, show_error=False)
