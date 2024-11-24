import spacy
from language_tool_python import LanguageTool


class GrammarChecker:
    def __init__(self):
        self.tool = LanguageTool('en-US')
        self.tool.disabled_rules.add('UPPERCASE_SENTENCE_START')  # Disable specific rule
        # Load the spaCy English language model once during initialization
        self.nlp = spacy.load("en_core_web_sm")
    
    def check_grammar(self, sentence: str) -> dict:
        """
        Checks if a string is grammatically correct and meaningful using spaCy and LanguageTool.

        Parameters:
            sentence (str): The input string to check.

        Returns:
            dict: A dictionary with grammaticality status and suggestions.
        """
        doc = self.nlp(sentence)

        # Check sentence parsing with spaCy
        if not doc.has_annotation("DEP"):
            return {"status": "Invalid", "message": "Sentence could not be parsed"}

        # Use LanguageTool for detailed grammar checking
        matches = self.tool.check(sentence)

        # Always run nonsensical checks
        is_nonsensical = self._is_nonsensical(doc)
        if is_nonsensical:
            return {
                "status": "Incorrect",
                "message": "The sentence appears nonsensical."
            }

        if matches:
            # Gather suggestions for improvement
            suggestions = [
                {
                    "error": match.context,
                    "suggestions": match.replacements,
                    "message": match.message
                }
                for match in matches
            ]

            return {
                "status": "Incorrect",
                "message": "The sentence has grammatical issues.",
                "issues": suggestions
            }

        # If no issues found
        return {"status": "Correct", "message": "The sentence is meaningful and correct."}

    def _is_nonsensical(self, doc) -> bool:
        """
        Perform semantic checks to detect nonsensical or incomplete sentences.

        Parameters:
            doc: A spaCy document object.

        Returns:
            bool: True if the sentence is nonsensical or incomplete.
        """
        # Detect meaningless repeated words (e.g., "the the the world")
        if any(doc[i].text.lower() == doc[i + 1].text.lower() for i in range(len(doc) - 1)):
            return True

        # Check if the sentence is a valid noun phrase
        root = [token for token in doc if token.dep_ == 'ROOT']
        if root and root[0].pos_ in {"NOUN", "PROPN"}:
            # Check if the noun phrase has a determiner or proper context
            if any(token.dep_ == "det" for token in doc):
                return False  # Valid noun phrase
            if len(doc) > 1 and all(token.pos_ in {"NOUN", "ADJ", "DET", "PROPN"} for token in doc):
                return False  # Likely a valid noun phrase

        # Check if the sentence contains at least one verb and one subject
        has_subject = any(token.dep_ in {"nsubj", "nsubjpass", "csubj"} for token in doc)
        has_verb = any(token.pos_ in {"VERB", "AUX"} for token in doc)

        if has_verb and any(token.dep_ == "expl" for token in doc):
            return False

        if has_verb and has_subject:
            # Check for a reasonable structure
            acceptable_deps = {
                "ROOT", "nsubj", "dobj", "pobj", "aux", "advmod", "prep", "det",
                "amod", "compound", "xcomp", "ccomp", "mark", "nmod", "attr", "acl",
                "advcl", "cc", "conj", "expl", "case", "poss", "npmod", "nummod", "agent"
            }
            unusual_deps = [token for token in doc if token.dep_ not in acceptable_deps]
            unusual_pos_sequences = [
                (doc[i].pos_, doc[i + 1].pos_) for i in range(len(doc) - 1)
                if (doc[i].pos_ in {"NOUN", "PROPN"} and doc[i + 1].pos_ == "VERB")
                or (doc[i].pos_ == "VERB" and doc[i + 1].pos_ in {"DET", "ADJ"})
                or (doc[i].pos_ == "CCONJ" and doc[i + 1].pos_ in {"VERB", "AUX"})
                or (doc[i].pos_ in {"NOUN", "PROPN"} and doc[i + 1].pos_ in {"CCONJ", "VERB"})
                or (doc[i].pos_ == "PRON" and doc[i + 1].pos_ in {"ADP", "DET", "CCONJ"} and doc[i].dep_ != "expl")
                or (doc[i].pos_ == "VERB" and doc[i + 1].pos_ == "ADP" and i + 2 < len(doc) and doc[i + 2].pos_ not in {"NOUN", "PROPN", "PRON"})
                or (doc[i].pos_ == "ADP" and doc[i + 1].pos_ == "DET" and (i + 2 == len(doc) or doc[i + 2].pos_ not in {"NOUN", "ADJ"}))
                or (doc[-1].pos_ == "DET")  # Check if the last token is a determiner (e.g., "a")
                or (doc[-1].pos_ == "ADP")  # Check if the last token is a preposition (e.g., "of")
                or (doc[i].pos_ == "DET" and doc[i + 1].pos_ in {"PRON", "DET"})  # Detect sequences with determiners followed by pronouns or another determiner
                or (doc[i].pos_ == "ADJ" and doc[i + 1].pos_ == "DET")  # Detect adjective followed by a determiner
                or (doc[i].pos_ == "NOUN" and doc[i + 1].pos_ == "ADV")  # Detect noun followed by an adverb
                or (doc[i].pos_ == "ADV" and doc[i + 1].pos_ in {"VERB", "AUX", "NOUN"})  # Detect adverb followed by inappropriate POS
                or (doc[-1].pos_ == "CCONJ")  # Check if the last token is a coordinating conjunction (e.g., "and")
                or (doc[-1].pos_ in {"PRON", "ADP", "AUX"})  # Check if the last token is a pronoun, preposition, or auxiliary verb (e.g., "he", "to", "are")
                or (doc[i].pos_ == "ADJ" and doc[i + 1].pos_ == "PRON")  # Detect adjective followed by a pronoun (e.g., "small thou")
                or (doc[i].pos_ == "NOUN" and doc[i + 1].pos_ == "PRON")  # Detect noun followed by a pronoun (e.g., "flower thou")
                or (doc[-1].text.lower() == "and")  # Ensure the last word cannot be "and"
            ]
            unusual_pos_sequences = []

            if len(unusual_deps) / len(doc) > 0.05 or len(unusual_pos_sequences) > 0:
                return True  # Too many unusual dependencies or uncommon POS sequences
            return False  # Acceptable structure

        # Sentence lacks a verb and isn't a valid noun phrase
        return True


## Example Usage
# checker = GrammarChecker()

# # Sentences to check
# sentences = [
#     "the wind of thing it",              # Nonsensical
#     "the wind of the world",             # Valid noun phrase
#     "this is a a a test",                # Repeated words
#     "wind the over",                     # Nonsensical
#     "I like to drink juice",             # Correct sentence
#     "the big green car",                 # Valid noun phrase
#     "the wind waves eat in",             # Nonsensical
#     "earth there than all the things clouds",  # Nonsensical
#     "the darkness in first",             # Nonsensical
#     "which others more",                 # Nonsensical
#     "a good place to die",               # Valid 
# ]

# # Check grammar
# for sentence in sentences:
#     result = checker.check_grammar(sentence)
#     print(f"Sentence: {sentence}")
#     print(result)
#     print()