from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

bot = ChatBot(
    "Tiberius",
    input_adapter="chatterbot_voice.VoiceInput",
    output_adapter="chatterbot_voice.VoiceOutput",
)

bot.set_trainer(ChatterBotCorpusTrainer)

# Train the chat bot with the entire english corpus
bot.train("chatterbot.corpus.english")

while True:
    try:
        # Use the parameter None because the VoiceInput adapter
        # is getting data from audio input instead of a parameter
        bot_input = bot.get_response(None)

    # Press ctrl-c or ctrl-d on the keyboard to exit
    except (KeyboardInterrupt, EOFError, SystemExit):
        break