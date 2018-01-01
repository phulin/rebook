from letters import Letter, TextLine

import numpy as np
cimport numpy as np

cimport cython

cdef class CLetter:
    cdef np.ndarray c
    cdef int x, y, w, h

    def __init__(self, letter):
        self.c = letter.c
        self.x = letter.x
        self.y = letter.y
        self.w = letter.w
        self.h = letter.h

    cpdef int r(self):
        return self.x + self.w

    cpdef int b(self):
        return self.y + self.h

def collate_lines(int AH, list letters):
    cdef int score, best_score, line_len
    cdef CLetter last1, last2, cl, letter
    cdef list lines, best_candidate

    letters = [CLetter(l) for l in letters]
    letters.sort(key=lambda CLetter cl: cl.x)

    lines = []
    for letter in letters:
        best_candidate = []
        best_score = 100000
        for line in lines:
            line_len = len(line)
            last1 = line[-1]
            last2 = line[-2] if line_len > 1 else last1
            score = best_score
            if letter.x < last1.r() + 4 * AH \
                    and last1.y <= letter.b() and letter.y <= last1.b():
                score = letter.x - last1.r() + abs(letter.y - last1.y)
            elif line_len > 1 \
                    and letter.x < last2.x + last2.w + AH \
                    and last2.y <= letter.b() and letter.y <= last2.b():
                score = letter.x - last2.r() + abs(letter.y - last2.y)
            if score < best_score:
                best_score = score
                best_candidate = line

        if best_candidate:
            best_candidate.append(letter)
            # print "  selected:", x, y, w, h
        else:
            lines.append([letter])

    return [TextLine([Letter(cl.c, cl.x, cl.y, cl.w, cl.h) for cl in line]) \
            for line in lines]
