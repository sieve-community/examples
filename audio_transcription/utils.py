import librosa
import numpy as np


def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000)
    return a


def load_audio_chunk(fname):
    audio = load_audio(fname)
    return audio


class HypothesisBuffer:
    def __init__(self):
        self.committed_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_committed_time = 0
        self.last_committed_word = None

    def insert(self, new, offset):
        # This is meant to set up self.new for the self.flush command to actually commit. Only insert words in new that extend the committed_in_buffer

        # Transform to absolute times
        new = [(start + offset, end + offset, word) for start, end, word in new]

        # Only consider words that are after the last committed word
        self.new = [val for val in new if val[0] > self.last_committed_time - 0.1]
        if len(self.new) > 0:
            start, _, _ = self.new[0]

            # If the new word is the same as the last committed word, then we might need to remove the prefix from self.new
            if abs(start - self.last_committed_time) < 1 and self.committed_in_buffer:
                # Search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in committed and new. If they are, they're dropped.
                cn = len(self.committed_in_buffer)
                nn = len(self.new)
                for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                    c = " ".join(
                        [self.committed_in_buffer[-j][2] for j in range(1, i + 1)][::-1]
                    )
                    tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                    if c == tail:
                        for _ in range(i):
                            self.new.pop(0)
                        break

    def flush(self):
        # Find the longest common prefix of 2 last inserts
        commit = []
        while self.new:
            ns, ne, nw = self.new[0]

            if len(self.buffer) == 0:
                break

            # If the words match between the buffer and the new, then we commit the word
            if nw == self.buffer[0][2]:
                commit.append((ns, ne, nw))
                self.last_committed_word = nw
                self.last_committed_time = ne
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(commit)
        return commit

    def pop_committed(self, time):
        while self.committed_in_buffer and self.committed_in_buffer[0][1] <= time:
            self.committed_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    SAMPLING_RATE = 16000

    def __init__(self, asr, tokenizer):
        """
        asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer.
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0
        self.transcript_buffer = HypothesisBuffer()
        self.committed = []
        self.last_chunked_at = 0
        self.silence_iters = 0

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of committed text that is inside of the scrolled away part of audio buffer.
        "context" is the committed text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0, len(self.committed) - 1)

        # TODO: CHANGED THIS TO > self.buffer_time_offset
        while k > 0 and self.committed[k - 1][1] > self.buffer_time_offset:
            k -= 1

        p = self.committed[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l < 200:
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
        non_prompt = self.committed[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(
            t for _, _, t in non_prompt
        )

    def process_iter(self):
        """
        Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
        The non-emty text is confirmed (committed) partial transcript.
        """

        prompt, non_prompt = self.prompt()
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        # TODO: just add this logic to res
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.committed.extend(o)

        # There's newly confirmed text
        if o:
            # We trim all the completed sentences from the audio buffer
            self.chunk_completed_sentence()

        # If the audio buffer is longer than 30s, trim it...
        if len(self.audio_buffer) / self.SAMPLING_RATE > 30:
            # ...on the last completed segment
            self.chunk_completed_segment(res)

        return self.to_flush(o)

    def chunk_completed_sentence(self):
        if self.committed == []:
            return
        sents = self.words_to_sentences(self.committed)
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        chunk_at = sents[-2][1]
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.committed == []:
            return

        all_segment_ends = self.asr.segments_end_ts(res)
        last_committed_end = self.committed[-1][1]

        if len(all_segment_ends) > 1:
            # We check 2 endings before, because we want something remaining after cut
            e = all_segment_ends[-2] + self.buffer_time_offset
            while len(all_segment_ends) > 2 and e > last_committed_end:
                all_segment_ends.pop(-1)
                e = all_segment_ends[-2] + self.buffer_time_offset

            # Trim at the last segment end before the last committed end
            if e <= last_committed_end:
                self.chunk_at(e)

    def chunk_at(self, time):
        self.transcript_buffer.pop_committed(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds) * self.SAMPLING_RATE :]
        self.buffer_time_offset = time
        self.last_chunked_at = time

    def words_to_sentences(self, words):
        """
        Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """

        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b, e, w = cwords.pop(0)
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg, end, fsent))
                    break
                sent = sent[len(w) :].strip()
        return out

    def finish(self):
        """
        Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        return f

    def to_flush(
        self,
        sents,
        sep=None,
        offset=0,
    ):
        # Concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1, end-of-last-sentence, "concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b, e, t)
