import argparse
import time
import random
import nltk
from nltk.corpus import wordnet
import multiprocessing
import os
import inflect
import pyinflect
import gensim.downloader as api
import pickle
import json
import re
import datetime

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize inflect engine
p = inflect.engine()

# Load or download word vectors
WORD_VECTORS_PATH = 'glove_vectors.pkl'
if os.path.exists(WORD_VECTORS_PATH):
    print("Loading saved word vectors...")
    with open(WORD_VECTORS_PATH, 'rb') as f:
        word_vectors = pickle.load(f)
else:
    print("Downloading word vectors... This may take a few minutes.")
    word_vectors = api.load('glove-wiki-gigaword-100')
    print("Saving word vectors for future use...")
    with open(WORD_VECTORS_PATH, 'wb') as f:
        pickle.dump(word_vectors, f)
print("Word vectors loaded.")

# Words to always skip (add any special tokens or words you want to preserve)
SKIP_WORDS = set(['endoftext'])

# Common words to avoid replacing
COMMON_WORDS = set(['do', 'does', 'did', 'done', 'be', 'is', 'am', 'are', 'was', 'were', 'been',
                    'have', 'has', 'had', 'go', 'goes', 'went', 'gone', 'make', 'makes', 'made',
                    'get', 'gets', 'got', 'gotten', 'the', 'a', 'an', 'and', 'or', 'but', 'in',
                    'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
                    'over', 'after', 'beneath', 'under', 'above'])

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def get_similar_synonyms(word, pos):
    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().lower()
            if synonym != word and '_' not in synonym:
                synonyms.add(synonym)

    synonyms = list(synonyms)
    if not synonyms:
        return []

    # Filter synonyms that are in our word vectors
    synonyms = [s for s in synonyms if s in word_vectors.key_to_index]

    if not synonyms:
        return synonyms

    # Calculate similarities
    try:
        similarities = [word_vectors.similarity(word, s) for s in synonyms]
    except KeyError:
        return []

    # Sort synonyms by similarity
    sorted_synonyms = [syn for _, syn in sorted(zip(similarities, synonyms), reverse=True)]

    return sorted_synonyms

def is_proper_noun(word, pos):
    return pos.startswith('NNP')

def apply_inflection(word, new_word, pos):
    if pos.startswith('NN'):
        if pos == 'NNS':
            return p.plural(new_word)
        return new_word
    elif pos.startswith('VB'):
        # Get the base form of the original word
        base_form = word.lower()
        if pos != 'VB':  # If it's not already in base form
            base_forms = pyinflect.getAllInflections(word)
            if base_forms and 'VB' in base_forms:
                base_form = base_forms['VB'][0]

        # Now inflect the new word
        inflections = pyinflect.getAllInflections(new_word)
        if inflections and pos in inflections:
            return inflections[pos][0]
        else:
            return new_word  # Return original if inflection not found
    return new_word

def count_eligible_words(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    eligible_words = 0
    total_words = 0

    for word, pos in pos_tags:
        total_words += 1
        if len(word) < 3 or word.lower() in SKIP_WORDS or word.lower() in COMMON_WORDS or is_proper_noun(word, pos):
            continue

        wordnet_pos = get_wordnet_pos(pos)
        if wordnet_pos not in [wordnet.ADJ, wordnet.ADV, wordnet.NOUN]:
            continue

        synonyms = get_similar_synonyms(word.lower(), wordnet_pos)
        if synonyms:
            eligible_words += 1

    return eligible_words, total_words

def replace_word(text, adjusted_replace_ratio, far_synonym_ratio):
    def replace_match(match):
        word = match.group(0)
        if len(word) < 3 or word.lower() in SKIP_WORDS or word.lower() in COMMON_WORDS:
            return word

        pos_tags = nltk.pos_tag([word])
        if not pos_tags:
            return word

        pos = pos_tags[0][1]
        if is_proper_noun(word, pos):
            return word

        wordnet_pos = get_wordnet_pos(pos)
        if wordnet_pos not in [wordnet.ADJ, wordnet.ADV, wordnet.NOUN]:
            return word

        synonyms = get_similar_synonyms(word.lower(), wordnet_pos)
        if not synonyms:
            return word

        if random.random() < adjusted_replace_ratio:
            if random.random() < far_synonym_ratio:
                new_word = random.choice(synonyms)
                is_far = True
            else:
                top_5 = synonyms[:5] if len(synonyms) >= 5 else synonyms
                new_word = random.choice(top_5)
                is_far = False

            new_word = apply_inflection(word, new_word, pos)

            # Preserve original capitalization
            if word.istitle():
                new_word = new_word.capitalize()
            elif word.isupper():
                new_word = new_word.upper()

            if word.lower() != new_word.lower():
                replacements.append((word, new_word, is_far))
                return new_word

        return word

    replacements = []
    close_synonyms = 0
    far_synonyms = 0

    new_text = re.sub(r'\b\w+\b', replace_match, text)

    for _, _, is_far in replacements:
        if is_far:
            far_synonyms += 1
        else:
            close_synonyms += 1

    return new_text, replacements, close_synonyms, far_synonyms

def count_words_in_chunk(chunk):
    eligible_words = 0
    total_words = 0
    for item in chunk:
        e, t = count_eligible_words(item['text'])
        eligible_words += e
        total_words += t
    return eligible_words, total_words

def augment_chunk(item, target_replace_ratio, far_synonym_ratio):
    augmented_text, replacements, close_synonyms, far_synonyms = replace_word(item['text'], target_replace_ratio, far_synonym_ratio)
    new_item = item.copy()
    new_item['text'] = augmented_text
    return new_item, replacements, close_synonyms, far_synonyms

def process_file(file_path, output_path, sample_size_mb=None, num_cores=None, target_replace_ratio=0.15, far_synonym_ratio=0.25):
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # If sample_size_mb is specified, take a subset of the data
    if sample_size_mb is not None:
        total_size = 0
        sample_data = []
        for item in data:
            total_size += len(json.dumps(item).encode('utf-8'))
            sample_data.append(item)
            if total_size >= sample_size_mb * 1024 * 1024:
                break
        data = sample_data

    # Split the data into chunks for multiprocessing
    chunk_size = max(1, len(data) // num_cores)
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    # First pass: count eligible words
    start_time = time.time()
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(count_words_in_chunk, chunks)

    total_eligible_words = sum(r[0] for r in results)
    total_words = sum(r[1] for r in results)

    # Adjust replacement ratio
    words_to_replace = int(total_words * target_replace_ratio)
    adjusted_replace_ratio = min(1.0, words_to_replace / total_eligible_words) if total_eligible_words > 0 else 0

    # Second pass: replace words
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(augment_chunk, [
            (item, adjusted_replace_ratio, far_synonym_ratio) for item in data
        ])
    end_time = time.time()

    # Collect all replacements, synonym counts, and augmented text
    all_replacements = []
    total_close_synonyms = 0
    total_far_synonyms = 0
    augmented_data = []
    for new_item, reps, close, far in results:
        augmented_data.append(new_item)
        all_replacements.extend(reps)
        total_close_synonyms += close
        total_far_synonyms += far

    processing_time = end_time - start_time

    # Save augmented text to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)

    # Calculate stats
    total_file_size = os.path.getsize(file_path) / (1024 * 1024)  # in MB

    return processing_time, total_file_size, all_replacements, total_close_synonyms, total_far_synonyms, total_eligible_words, total_words, adjusted_replace_ratio

def log_replacements(replacements, log_file):
    with open(log_file, 'w', encoding='utf-8') as f:
        for original, replacement, is_far in replacements:
            far_indicator = " (far)" if is_far else ""
            f.write(f"{original} -> {replacement}{far_indicator}\n")

def log_stats(log_file, stats):
    with open(log_file, 'w', encoding='utf-8') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Augmentation Script")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_dir", help="Directory to save all output files")
    parser.add_argument("--sample_size", type=float, help="Size of sample to process in MB. If not specified, processes entire file.")
    parser.add_argument("--num_cores", type=int, default=None, help="Number of CPU cores to use")
    parser.add_argument("--replace_ratio", type=float, default=0.1, help="Proportion of words to attempt to replace (0-1)")
    parser.add_argument("--far_ratio", type=float, default=0.3, help="Proportion of replacements that should use 'far' synonyms (0-1)")
    args = parser.parse_args()

    # Ensure ratios are between 0 and 1
    args.replace_ratio = min(1.0, max(0.0, args.replace_ratio))
    args.far_ratio = min(1.0, max(0.0, args.far_ratio))

    # Ensure output directory exists
    ensure_dir(args.output_dir)

    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define output file paths
    output_file = os.path.join(args.output_dir, "output.json")
    replacements_log = os.path.join(args.output_dir, "replacements.log")
    stats_log = os.path.join(args.output_dir, "stats.log")

    processing_time, total_file_size, replacements, close_count, far_count, eligible_words, total_words, adjusted_replace_ratio = process_file(
        args.input_file, output_file, args.sample_size, args.num_cores, args.replace_ratio, args.far_ratio
    )

    # Always log replacements
    log_replacements(replacements, replacements_log)

    total_replacements = close_count + far_count
    actual_replace_ratio = total_replacements / total_words if total_words > 0 else 0
    actual_far_ratio = far_count / total_replacements if total_replacements > 0 else 0

    stats = {
        "Sample size processed (MB)": f"{total_file_size:.2f}",
        "Processing time (seconds)": f"{processing_time:.2f}",
        "Cores used": args.num_cores,
        "Total words": total_words,
        "Eligible words": eligible_words,
        "Adjusted replacement ratio": f"{adjusted_replace_ratio:.2%}",
        "Total number of replacements": total_replacements,
        "Actual replacement rate": f"{actual_replace_ratio:.2%}",
        "Target replacement rate": f"{args.replace_ratio:.2%}",
        "Close synonyms": close_count,
        "Far synonyms": far_count,
        "Actual proportion of far synonyms": f"{actual_far_ratio:.2%}",
        "Target proportion of far synonyms": f"{args.far_ratio:.2%}",
        "Augmented text saved to": output_file,
        "Replacement log saved to": replacements_log
    }

    log_stats(stats_log, stats)

    print(f"Augmentation complete. Files saved in {args.output_dir}:")
    print(f"  Augmented text: {os.path.basename(output_file)}")
    print(f"  Replacements log: {os.path.basename(replacements_log)}")
    print(f"  Statistics log: {os.path.basename(stats_log)}")