import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

config = dotenv_values(".env")
API_KEY_ELEVEN = config["API_KEY_ELEVEN"]

def translator(audio_file):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language="Spanish", fp16=False)
        transcription = result["text"]
    except Exception as e:
        raise gr.Error(gr.Error(f"se ha producido un error transcribiendo {str(e)}"))
    

    try:
        en_translator = Translator(from_lang="es", to_lang="en").translate(transcription)
        cn_translator = Translator(from_lang="es", to_lang="zh").translate(transcription)
        jp_translator = Translator(from_lang="es", to_lang="ja").translate(transcription)
        kr_translator = Translator(from_lang="es", to_lang="ko").translate(transcription)
    except Exception as e:
        raise gr.Error(gr.Error(f"se ha producido un error traduciendo {str(e)}"))
    
    en_save_file_path = text_to_speach(en_translator, "en")
    cn_save_file_path = text_to_speach(cn_translator, "cn")
    jp_save_file_path = text_to_speach(jp_translator, "jp")
    kr_save_file_path = text_to_speach(kr_translator, "kr")
    
    return en_save_file_path, cn_save_file_path, jp_save_file_path, kr_save_file_path
    
def text_to_speach(text: str, language: str) -> str:
    client = ElevenLabs(api_key=API_KEY_ELEVEN)
    try:
        response = client.text_to_speech.convert(
                voice_id="EXAVITQu4vr4xnSDxMaL", #sarah
                optimize_streaming_latency="0",
                output_format="mp3_22050_32",
                text=text,
                model_id="eleven_turbo_v2",
                voice_settings=VoiceSettings(
                    stability=0.0,
                    similarity_boost=1.0,
                    style=0.0,
                    use_speaker_boost=True,
                ),
            )
        
        save_file_path = f"audios/{language}.mp3"

        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)
    except Exception as e:
        raise gr.Error(gr.Error(f"se ha producido un error creando el audio {str(e)}"))

    return save_file_path

web = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Español"
    ),
    outputs=[
        gr.Audio(label="Inglés"),
        gr.Audio(label="Chino"),
        gr.Audio(label="Japonés"),
        gr.Audio(label="Coreano")
    ],
    title="DemoTranslator",
    description="El traductor de los rial demoños"
)

web.launch()
