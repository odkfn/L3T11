import spacy
nlp = spacy.load("en_core_web_sm")

# Define 5 garden path sentences
sentence_one = "The old man the boat."
sentence_two = "The horse raced past the barn fell."
sentence_three = "The complex houses married and single soldiers and their families."
sentence_four = "We painted the wall with cracks."
sentence_five = "The sour drink from the ocean."
# I wasn't getting any entity recognition for the five sentences so I added
# The example sentence from the tutorial to ensure that my code was not the
# issue.
sentence_six = """known by her married name Priyanka Chopra Jonas, is an Indian actress,
singer, film producer, philanthropist, and the winner of the Miss World 2000 pageant.
One of India's highest-paid and most popular celebrities, Chopra has received numerous
awards, including a National Film Award and five Filmfare Awards. In 2016, the Government
of India honoured her with the Padma Shri, and Time named her one of the 100 most influential people in the world."""

# Add the sentences to a list
gardenpath_sentences = [sentence_one, sentence_two, sentence_three, sentence_four, sentence_five, sentence_six]

# Loop through the list, tokenising each and perform entity recognition.

for sentence in gardenpath_sentences:
    tokenised_sentence = nlp(sentence)
    print([token.orth_ for token in tokenised_sentence if not token.is_punct | token.is_space])
    print([(ent.text, ent.label_) for ent in tokenised_sentence.ents])
    
# It seems that garden path sentences do not yield any entities.  This was tested and proven by including sentence_six
# aka the sentence from the tutorial, which did yield results.  I also tested this by changing sentence five briefly to
# "The sour in the UK drink from the ocean" and the UK was labelled as GPE which is a geopolitical entity. i.e. a country.
