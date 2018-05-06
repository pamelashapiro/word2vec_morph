//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

struct lemma_struct {
  long long cn;
  char *lemma;
};

struct lemma_count {
  long long lemma;
  long long cn;
  struct lemma_count *next;
};

char train_file[MAX_STRING], output_file[MAX_STRING], output_lemmas_file[MAX_STRING], output_num_lemmas_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
char save_lemmas_file[MAX_STRING], read_lemmas_file[MAX_STRING];
struct vocab_word *vocab;
struct lemma_struct *lemmas;
struct lemma_count *word_lemma_counts;

int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
int *lemma_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_w_size = 50;
long long lemmas_max_size = 1000, lemmas_size = 0, layer1_l_size = 50; 
long long layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0_w, *syn0_l, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

struct lemma_count* CreateNode(long long lemma, struct lemma_count* next)
{
  struct lemma_count* new_node = (struct lemma_count*)calloc(sizeof(struct lemma_count),1);
  if(new_node == NULL)
  {
      printf("Error creating a new node in linked list.\n");
      exit(-1);
  }
  new_node->lemma = lemma;
  new_node->cn = 1;
  new_node->next = next;

  return new_node;
}

struct lemma_count* AppendNode(struct lemma_count* head, long long lemma)
{
    struct lemma_count *cursor = head;
    while(cursor->next != NULL)
        cursor = cursor->next;

    struct lemma_count* new_node =  CreateNode(lemma,NULL);
    cursor->next = new_node;
 
    return head;
}

void FindAndIncrementNode(struct lemma_count* head, long long lemma) {

  struct lemma_count* cursor = head;
  while(cursor != NULL)
  {
      if (cursor->lemma == lemma) {
        cursor->cn++;
        return;
      }
      cursor = cursor->next;
  }

  head = AppendNode(head, lemma);
  return;
}

int NumLemmas(struct lemma_count* head) {
    int num = 0;
    struct lemma_count* cursor = head;
    while (cursor != NULL)
    {
        num += 1;
        cursor = cursor->next;
    }

    return num;
}

void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin, char *eof) {
  int a = 0, ch;
  while (1) {
    ch = fgetc_unlocked(fin);
    if (ch == EOF) {
      *eof = 1;
      break;
    }
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Reads a single word and its lemma from a file, assuming space + tab + EOL to be word boundaries,
// word and lemma separated by a colon
void ReadWordLemma(char *word, char *lemma, FILE *fin, char *eof) {
  int a = 0, b = 0, ch;
  int reached_colon = 0;
  int added_lemma = 0;
  while (1) {
    ch = fgetc_unlocked(fin);
    if (ch == EOF) {
      *eof = 1;
      break;
    }
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        strcpy(lemma, (char *)"</s>");
        return;
      } else continue;
    }
    if (ch == ':') {
      reached_colon = 1;
      continue;
    }
    if (!reached_colon) {
      word[a] = ch;
      a++;
      if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    } else {
      added_lemma = 1;
      lemma[b] = ch;
      b++;
      if (b >= MAX_STRING - 1) b--;
    }
  }
  word[a] = 0;
  if (!reached_colon || !added_lemma) {
    strcpy(lemma, word);
  }
  else {
    lemma[b] = 0;
  }
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns hash value of a lemma
int GetLemmaHash(char *lemma) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(lemma); a++) hash = hash * 257 + lemma[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Returns position of a lemma in the vocabulary; if the lemma is not found, returns -1
int SearchLemmas(char *lemma) {
  unsigned int hash = GetLemmaHash(lemma);
  while (1) {
    if (lemma_hash[hash] == -1) return -1;
    if (!strcmp(lemma, lemmas[lemma_hash[hash]].lemma)) return lemma_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and its lemma and returns indices in the vocabulary and lemmas
int ReadWordLemmaIndices(FILE *fin, char *eof, long long *indices) {
  char word[MAX_STRING], lemma[MAX_STRING], eof_l = 0;
  ReadWordLemma(word, lemma, fin, &eof_l);
  if (eof_l) {
    *eof = 1;
    return -1;
  }

  indices[0] = SearchVocab(word);
  indices[1] = SearchLemmas(lemma);

  return 0;
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Adds a lemma to lemmas
int AddLemmaToLemmas(char *lemma) {
  unsigned int hash, length = strlen(lemma) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  lemmas[lemmas_size].lemma = (char *)calloc(length, sizeof(char));
  strcpy(lemmas[lemmas_size].lemma, lemma);
  lemmas_size++;
  // Reallocate memory if needed
  if (lemmas_size + 2 >= lemmas_max_size) {
    lemmas_max_size += 1000;
    lemmas = (struct lemma_struct *)realloc(lemmas, lemmas_max_size * sizeof(struct lemma_struct));
  }
  hash = GetLemmaHash(lemma);
  while (lemma_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  lemma_hash[hash] = lemmas_size - 1;
  return lemmas_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  long long l = ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
  if (l > 0) return 1;
  if (l < 0) return -1;
  return 0;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void ReduceLemmas() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < lemmas_size; a++) if (lemmas[a].cn > min_reduce) {
    lemmas[b].cn = lemmas[a].cn;
    lemmas[b].lemma = lemmas[a].lemma;
    b++;
  } else free(lemmas[a].lemma);
  lemmas_size = b;
  for (a = 0; a < vocab_hash_size; a++) lemma_hash[a] = -1;
  for (a = 0; a < lemmas_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetLemmaHash(lemmas[a].lemma);
    while (lemma_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    lemma_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabLemmasFromTrainFile() {

  char word[MAX_STRING], lemma[MAX_STRING], eof = 0;
  FILE *fin;
  long long a, i, b, j, wc = 0;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_hash_size; a++) lemma_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  lemmas_size = 0;
  AddWordToVocab((char *)"</s>");
  AddLemmaToLemmas((char *)"</s>");
  while (1) {
    ReadWordLemma(word, lemma, fin, &eof);
    if (eof) break;
    train_words++;
    wc++;
    /*if (wc % 10000 == 0) {
      printf("read %lld words into vocab\n", wc);
    }*/
    if ((debug_mode > 1) && (wc >= 1000000)) {
      printf("%lldM%c", train_words / 1000000, 13);
      fflush(stdout);
      wc = 0;
    }
    i = SearchVocab(word);
    j = SearchLemmas(lemma);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (j == -1) {
      b = AddLemmaToLemmas(lemma);
      lemmas[b].cn = 1;
    } else lemmas[j].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    if (lemmas_size > vocab_hash_size * 0.7) ReduceLemmas();
  }
  SortVocab();
  // Should I do this for lemmas? Not sure what the purpose is
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Number of lemmas: %lld\n", lemmas_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

// Should probably do this for lemmas, but not necessary right now

void SaveVocabAndLemmas() {
  long long i;
  FILE *fo_v, *fo_l;
  fo_v = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo_v, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo_v);

  fo_l = fopen(save_lemmas_file, "wb");
  for (i = 0; i < lemmas_size; i++) fprintf(fo_l, "%s %lld\n", lemmas[i].lemma, lemmas[i].cn);
  fclose(fo_l);
}

void ReadVocabAndLemmas() {
  long long a, i = 0;
  char c, eof = 0;
  char word[MAX_STRING], lemma[MAX_STRING];
  FILE *fin;
  fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }

  fin = fopen(read_lemmas_file, "rb");
  if (fin == NULL) {
    printf("Lemma file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) lemma_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(lemma, fin, &eof);
    if (eof) break;
    a = AddLemmaToLemmas(lemma);
    fscanf(fin, "%lld%c", &lemmas[a].cn, &c);
    i++;
  }
  fclose(fin);

  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0_w, 128, (long long)vocab_size * layer1_w_size * sizeof(real));
  if (syn0_w == NULL) {printf("Memory allocation failed\n"); exit(1);}
  b = posix_memalign((void **)&syn0_l, 128, (long long)lemmas_size * layer1_l_size * sizeof(real));
  if (syn0_l == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_w_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0_w[a * layer1_w_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_w_size;
  }
  for (a = 0; a < lemmas_size; a++) for (b = 0; b < layer1_l_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0_l[a * layer1_l_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_l_size;
  }
  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, lemma, last_word, last_lemma, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen_w[MAX_SENTENCE_LENGTH + 1], sen_l[MAX_SENTENCE_LENGTH + 1];
  long long l1_w, l1_l, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  char eof = 0;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        long long indices[2];
        indices[0] = -1;
        indices[1] = -1;
        ReadWordLemmaIndices(fi, &eof, indices);
        word = indices[0];
        lemma = indices[1];
        if (eof) break;
        if (word == -1) continue;

        word_count++;

        if (lemma == -1) continue;

        //Add to word_lemma_counts here
        if (word_lemma_counts[word].lemma == -1) {
          word_lemma_counts[word].lemma = lemma;
          word_lemma_counts[word].cn = 1;
        } else {
          FindAndIncrementNode(&(word_lemma_counts[word]), lemma);
        }

        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }

        sen_w[sentence_length] = word;
        sen_l[sentence_length] = lemma;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (eof || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      eof = 0;
      continue;
    }
    word = sen_w[sentence_position];
    lemma = sen_l[sentence_position];
    if (word == -1 || lemma == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      fprintf(stderr, "%s", "CBOW is not implented yet\n");
      exit(-1);
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a; // c is index in sentence of target word
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen_w[c];
        last_lemma = sen_l[c];
        if (last_word == -1 || last_lemma) continue;
        // Add up the elements of the input vector for a word
        for (c = 0; c < layer1_w_size; c++) neu1[c] += syn0_w[c + last_word * layer1_w_size];
        for (c = 0; c < layer1_l_size; c++) neu1[layer1_w_size + c] += syn0_l[c + last_lemma * layer1_l_size];
        cw++;
      }
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen_w[c];
          last_lemma = sen_l[c];
          if (last_word == -1 || last_lemma == -1) continue; // not sure if I want to check both
          for (c = 0; c < layer1_w_size; c++) syn0_w[c + last_word * layer1_w_size] += neu1e[c];
          for (c = 0; c < layer1_l_size; c++) syn0_l[c + last_lemma * layer1_l_size] += neu1e[layer1_w_size + c];
        }
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen_w[c];
        last_lemma = sen_l[c];
        if (last_word == -1 || last_lemma == -1) continue; // not sure if I want to check bot
        l1_w = last_word * layer1_w_size;
        l1_l = last_lemma * layer1_l_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          fprintf(stderr, "%s", "Hierarchical Softmax is not implented yet\n");
          exit(-1);
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_w_size; c++) f += syn0_w[c + l1_w] * syn1[c + l2];
          for (c = 0; c < layer1_l_size; c++) f += syn0_l[c + l1_l] * syn1[layer1_w_size + c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_w_size; c++) syn1[c + l2] += g * syn0_w[c + l1_w];
          for (c = 0; c < layer1_l_size; c++) syn1[layer1_w_size + c + l2] += g * syn0_l[c + l1_l];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0; // dot product
          for (c = 0; c < layer1_w_size; c++) f += syn0_w[c + l1_w] * syn1neg[c + l2];
          for (c = 0; c < layer1_l_size; c++) f += syn0_l[c + l1_l] * syn1neg[layer1_w_size + c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_w_size; c++) syn1neg[c + l2] += g * syn0_w[c + l1_w];
          for (c = 0; c < layer1_l_size; c++) syn1neg[layer1_w_size + c + l2] += g * syn0_l[c + l1_l];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_w_size; c++) syn0_w[c + l1_w] += neu1e[c];
        for (c = 0; c < layer1_l_size; c++) syn0_l[c + l1_l] += neu1e[layer1_w_size + c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo, *fo_l, *fo_num_l;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  real *lemma_average = (real *)calloc(layer1_l_size, sizeof(real));
  struct lemma_count* cursor;
  double avg_denom;
  long long lemma_index, num_lemmas;
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0  && read_lemmas_file != 0) ReadVocabAndLemmas(); else LearnVocabLemmasFromTrainFile();
  if (save_vocab_file[0] != 0 && save_lemmas_file != 0) SaveVocabAndLemmas();
  if (output_file[0] == 0 || output_lemmas_file[0] == 0 || output_num_lemmas_file[0] == 0) {
    printf("Skipping model training because an output file was missing.\n");
    return;
  }
  word_lemma_counts = (struct lemma_count *)calloc(vocab_size, sizeof(struct lemma_count));
  for (a = 0; a < vocab_size; a++) word_lemma_counts[a].lemma = -1;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  fo_num_l = fopen(output_num_lemmas_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld %lld\n", vocab_size, layer1_size, layer1_w_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      fprintf(fo_num_l, "%s ", vocab[a].word);
      if (binary) {
        for (b = 0; b < layer1_w_size; b++) fwrite(&syn0_w[a * layer1_w_size + b], sizeof(real), 1, fo);
        // want to average lemma vectors here
        cursor = &word_lemma_counts[a];
        avg_denom = 0;
        num_lemmas = 0; 
        for (b = 0; b < layer1_l_size; b++) lemma_average[b] = 0; 
        while(cursor != NULL)
        {
            lemma_index = cursor->lemma;
            for (b = 0; b < layer1_l_size; b++) lemma_average[b] += cursor->cn * syn0_l[lemma_index*layer1_l_size + b];
            avg_denom += cursor->cn;
            num_lemmas++;
            cursor = cursor->next;
        }
	if (avg_denom != 0) {
	  for (b = 0; b < layer1_l_size; b++) lemma_average[b] /= avg_denom;
	}
        for (b = 0; b < layer1_l_size; b++) fwrite(&lemma_average[b], sizeof(real), 1, fo);
      }
      else {
        for (b = 0; b < layer1_w_size; b++) fprintf(fo, "%lf ", syn0_w[a * layer1_w_size + b]);
        // want to average lemma vectors here
        cursor = &word_lemma_counts[a];
        avg_denom = 0;
        num_lemmas = 0; 
        for (b = 0; b < layer1_l_size; b++) lemma_average[b] = 0;  
        while(cursor != NULL)
        {
            lemma_index = cursor->lemma;
            for (b = 0; b < layer1_l_size; b++) lemma_average[b] += cursor->cn * syn0_l[lemma_index*layer1_l_size + b];
            avg_denom += cursor->cn;
            num_lemmas++;
            cursor = cursor->next;
        }

        fprintf(fo_num_l, " %lld\n", num_lemmas);

	if (avg_denom != 0) {
	  for (b = 0; b < layer1_l_size; b++) lemma_average[b] /= avg_denom;
	}
        for (b = 0; b < layer1_l_size; b++) fprintf(fo, "%lf ", lemma_average[b]);
      }
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    fprintf(stderr, "%s", "K-means is not implented yet\n");
    exit(-1);
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_w_size; d++) cent[layer1_size * cl[c] + d] += syn0_w[c * layer1_w_size + d];
        for (d = 0; d < layer1_l_size; d++) cent[layer1_size * cl[c] + d] += syn0_l[c * layer1_l_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_w_size; b++) x += cent[layer1_size * d + b] * syn0_w[c * layer1_w_size + b];
          for (b = 0; b < layer1_l_size; b++) x += cent[layer1_size * d + b] * syn0_l[c * layer1_l_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
  fclose(fo_num_l);


  fo_l = fopen(output_lemmas_file, "wb");
  if (classes == 0) {
    // Save the lemmas vectors
    fprintf(fo_l, "%lld %lld\n", lemmas_size, layer1_l_size);
    for (a = 0; a < lemmas_size; a++) {
      fprintf(fo_l, "%s ", lemmas[a].lemma);
      if (binary) {
        for (b = 0; b < layer1_l_size; b++) fwrite(&syn0_l[a * layer1_l_size + b], sizeof(real), 1, fo_l);
      }
      else {
        for (b = 0; b < layer1_l_size; b++) fprintf(fo_l, "%lf ", syn0_l[a * layer1_l_size + b]);
      }
      fprintf(fo_l, "\n");
    }
  }
  // Don't bother with else clause for now, might want to implement later 
  fclose(fo_l);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c, modified to take in word:lemma\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-output-lemmas <file>\n");
    printf("\t\tUse <file> to save the resulting lemma vectors\n");
    printf("\t-word-size <int>\n");
    printf("\t\tSet size of word vectors; default is 50\n");
    printf("\t-lemma-size <int>\n");
    printf("\t\tSet size of lemma vectors; default is 50\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-save-lemmas <file>\n");
    printf("\t\tThe lemmas will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-read-lemmas <file>\n");
    printf("\t\tThe lemmas will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec_morph -train data.txt -output vec.txt -output-lemmas vec_l.txt -word-size 100 -lemma-size 100 -window 5 -sample 1e-4 -negative 5 -hs 1 -binary 0 -cbow 0 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  output_lemmas_file[0] = 0;
  output_num_lemmas_file[0] = 0;
  save_vocab_file[0] = 0;
  save_lemmas_file[0] = 0;
  read_vocab_file[0] = 0;
  read_lemmas_file[0] = 0;
  if ((i = ArgPos((char *)"-word-size", argc, argv)) > 0) layer1_w_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-lemma-size", argc, argv)) > 0) layer1_l_size = atoi(argv[i + 1]);
  layer1_size = layer1_w_size + layer1_l_size;
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-lemmas", argc, argv)) > 0) strcpy(save_lemmas_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-lemmas", argc, argv)) > 0) strcpy(read_lemmas_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output-lemmas", argc, argv)) > 0) strcpy(output_lemmas_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output-num-lemmas", argc, argv)) > 0) strcpy(output_num_lemmas_file, argv[i+1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  lemmas = (struct lemma_struct *)calloc(lemmas_max_size, sizeof(struct lemma_struct));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  lemma_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
