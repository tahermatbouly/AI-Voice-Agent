from voicetut_tts import VoiceTutTTS

tts = VoiceTutTTS.from_pretrained("mohammedaly22/VoiceTut-TTS")

# 1) Built-in speaker
tts.synthesize("ازيك عامل ايه النهاردة؟", speaker="Mohamed", output="out.wav")

# 2) Zero-shot voice cloning
tts.synthesize("النهارده الجو حلو اوي",
               ref_audio="my_voice.wav", ref_text="ده الصوت بتاعي", output="clone.wav")

# 3) Code-switching + generation params
tts.synthesize("عندي meeting الساعة 3:30 ومعايا ال presentation",
               speaker="Asmaa", num_step=48, guidance_scale=2.5, speed=1.05, output="cs.wav")
