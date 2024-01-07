import concurrent.futures
import re
import nltk
import spacy
import torch
import logging
from nltk import word_tokenize, pos_tag, sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
import uuid
import numpy as np
from chunkipy import TextChunker
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bark import SAMPLE_RATE, generate_audio, preload_models
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from summa import summarizer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def download_nltk_data():
   try:
       nltk.download('punkt')
       nltk.download('averaged_perceptron_tagger')
       nltk.download('vader_lexicon')
       logging.info("NLTK data downloaded successfully.")
   except Exception as e:
       logging.error(f"Error downloading NLTK data: {e}")

class TextProcessor:
   def __init__(self):
       try:
           self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True)
           self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
           self.tokenizer.add_special_tokens({'pad_token': '[MULTIVERSETUNE|X:34|Y:76Z|12|T:5633]'})
           self.model.to("cuda")
       except Exception as e:
           logging.error(f"Error initializing TextProcessor: {e}")
           raise
       
   def play_response_audio(self, response_text):
        try:
            sentences = re.split('(?<=[.!?]) +', response_text)
            silence = np.zeros(int(0.05 * SAMPLE_RATE))

            def generate_sentence_audio(sentence):
                try:
                    return generate_audio(sentence, history_prompt="v2/en_speaker_6")
                except Exception as e:
                    logging.error(f"Error generating audio for sentence '{sentence}': {e}")
                    return np.zeros(0)

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(sentences))) as executor:
                audio_arrays = list(executor.map(generate_sentence_audio, sentences))

            audio_arrays = [audio for audio in audio_arrays if audio.size > 0]

            if audio_arrays:
                pieces = [piece for audio in audio_arrays for piece in (audio, silence.copy())]
                audio = np.concatenate(pieces[:-1])

                file_name = str(uuid.uuid4()) + ".wav"
                write_wav(file_name, SAMPLE_RATE, audio)
                sd.play(audio, samplerate=SAMPLE_RATE)
            else:
                logging.error("No audio generated due to errors in all sentences.")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Error in play_response_audio: {e}")
   def is_code_like(self, chunk):
       try:
           code_patterns = r'\b(def|class|import|if|else|for|while|return|function|var|let|const|print)\b|[\{\}\(\)=><\+\-\*/]'
           return bool(re.search(code_patterns, chunk))
       except Exception as e:
           logging.error(f"Error in is_code_like: {e}")
           return False

   def text_ends_incomplete(self, text):
       try:
           if not re.search(r'[.?!]\s*$', text):
               return True

           brackets = {'(': ')', '{': '}', '[': ']'}
           stack = []
           for char in text:
               if char in brackets:
                  stack.append(char)
               elif char in brackets.values():
                  if not stack or brackets[stack.pop()] != char:
                      return True

           return bool(stack)
       except Exception as e:
           logging.error(f"Error in text_ends_incomplete: {e}")
           return True

   def calculate_lexical_density(self, text):
       try:
           content_pos_tags = {'NN', 'VB', 'JJ', 'RB'}
           words = word_tokenize(text)
           content_words = [word for word, tag in pos_tag(words) if tag[:2] in content_pos_tags]
           return len(content_words) / len(words) if words else 0
       except Exception as e:
           logging.error(f"Error in calculate_lexical_density: {e}")
           return 0

   def calculate_syntactic_complexity(self, text):
       try:
           doc = spacy.load("en_core_web_sm")(text)
           long_sentences = sum(1 for sent in doc.sents if len(sent) > 15)
           subordinate_clauses = sum(1 for token in doc if token.dep_ in {"ccomp", "xcomp"})
           passive_voice = sum(1 for token in doc if token.tag_ in {"VBN", "VBD"} and token.dep_ == "auxpass")
           return long_sentences + subordinate_clauses + passive_voice
       except Exception as e:
           logging.error(f"Error in calculate_syntactic_complexity: {e}")
           return 0

   def determine_max_chunk_size(self, text, base_size=3, density_threshold=0.6, complexity_threshold=5):
       try:
           density = self.calculate_lexical_density(text)
           complexity = self.calculate_syntactic_complexity(text)
           if density > density_threshold or complexity > complexity_threshold:
               return max(1, base_size - 1)
           return base_size
       except Exception as e:
           logging.error(f"Error in determine_max_chunk_size: {e}")
           return base_size

   def split_into_chunks(self, text):
       try:
           text_chunker = TextChunker(chunk_size=800, tokens=True, overlap_percent=0.6)
           chunks = text_chunker.chunk(text)
           return chunks
       except Exception as e:
           logging.error(f"Error in split_into_chunks: {e}")
           return []

   def structural_analysis(self, text):
       try:
           doc = spacy.load("en_core_web_sm")(text)
           sentence_types = {"interrogative": False, "imperative": False, "declarative": False}
           for sent in doc.sents:
               if sent.text.endswith("?"):
                  sentence_types["interrogative"] = True
               elif sent[0].tag_ in ["VB", "MD"]:
                  sentence_types["imperative"] = True
               else:
                  sentence_types["declarative"] = True
           return sentence_types
       except Exception as e:
           logging.error(f"Error in structural_analysis: {e}")
           return {"interrogative": False, "imperative": False, "declarative": False}

   def dynamic_token_creation(self, text, sentiment=""):
      try:
          sid = SentimentIntensityAnalyzer()
          sentiment = sid.polarity_scores(text)
          structure = self.structural_analysis(text)
          tokens = []

          if structure["interrogative"]:
              tokens.append("{{{question}}}")
          if structure["imperative"]:
              tokens.append("{{{command}}}")
          if structure["declarative"]:
              tokens.append("{{{statement}}}")

          tokens.append(f"{{{sentiment}}}")
          return ' '.join(tokens) + " " + text
      except Exception as e:
          logging.error(f"Error in dynamic_token_creation: {e}")
          return text
   def process_text(self, text, sentiment=""):
      try:
          if self.is_code_like(text):
              return "[code] " + text
          return self.dynamic_token_creation(text, sentiment=sentiment)
      except Exception as e:
          logging.error(f"Error in process_text: {e}")
          return text

   def generate_text(self, input_text, max_length=1200):
      try:
          inputs = self.tokenizer(input_text, return_tensors="pt", return_attention_mask=True, padding=True, truncation=True)
          inputs = {key: value.to("cuda") for key, value in inputs.items()} 
          outputs = self.model.generate(**inputs, max_length=max_length, return_dict_in_generate=True)
          generated_ids = outputs.sequences
          generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
          return generated_text
      except Exception as e:
          logging.error(f"Error in generate_text: {e}")
          return ""

   def run_in_thread(self, func, *args, **kwargs):
      thread = threading.Thread(target=func, args=args, kwargs=kwargs)
      thread.start()
      thread.join()

def main():
    processor = TextProcessor()

    text = "USERASKS hi, write a 15000 word story about growing android life forms bioandroids using advanced quantum processes of communication, use two generation agents"
    chunks = processor.split_into_chunks(text)

    generated_texts = []  # List to store generated texts

    def process_chunk_and_play_audio(chunk, sentiment):
        processed_text = processor.process_text(chunk, sentiment=sentiment)
        generated_text = processor.generate_text(processed_text)
        generated_texts.append(generated_text)  # Append generated text to the list
        print(f"Chunk: {chunk}\nGenerated: {generated_text}\n")
        processor.play_response_audio(generated_text)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each chunk for processing and audio playback
        futures = [executor.submit(process_chunk_and_play_audio, chunk, f"sentiment_{i}") for i, chunk in enumerate(chunks)]

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    # Output all generated text in the console
    combined_text = ' '.join(generated_texts)
    print("\nAll Generated Text:\n", combined_text)

if __name__ == "__main__":
    main()
