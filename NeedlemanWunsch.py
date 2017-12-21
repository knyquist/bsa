import numpy as np

class GlobalAlignment:
    """
    Global aligner for nucleotide sequences.
    Uses Needleman-Wunsch with linear gap penalty.
    """
    def __init__(self, seq1, seq2):
        self.seq1 = seq1
        self.seq2 = seq2
        self.b2i_map = self.b2i_map() 
        self.S = self.substitution_matrix()
        self.score, self.trace = self.align()
        self.alignment = self.optimal_alignment()
    
    def b2i_map(self):
        """
        Dictionary mapping base to index
        """
        return {'A': 0,
                'C': 1,
                'G': 2,
                'T': 3}
    
    def substitution_matrix(self):
        """
        Substitution matrix
        4x4 matrix, base order ACTG left/right, top/bottom
        """
        S = np.array([[1, -1, -1, -1],
                      [-1, 1, -1, -1],
                      [-1, -1, 1, -1],
                      [-1, -1, -1, 1]])
        return S
    
    def b2i(self, base):
        """
        Base to index. Enables easy access to substitution
        matrix
        """
        return self.b2i_map[base]
    
    def S_w(self, b1, b2):
        """
        Retrieve weight from substitution matrix
        for base b1 and base b2
        """
        return self.S[self.b2i(b1), self.b2i(b2)]
    
    def G_w(self):
        """
        Retrieve gap penalty weight.
        For now, use simple linear penalty
        """
        return -1
    
    def initialize_score_matrix(self, score_matrix):
        """
        Fill the first row and column with gap penalties
        """
        score_matrix[0, 0] = 0
        for i in np.arange(1, score_matrix.shape[0]):
            score_matrix[i, 0] = score_matrix[i-1, 0] + self.G_w()
        for j in np.arange(1, score_matrix.shape[1]):
            score_matrix[0, j] = score_matrix[0, j-1] + self.G_w()
        return score_matrix
    
    def fill_score_ij(self, score_matrix, i, j):
        """
        Return max score for element i,j given previous scores
        """
        diag = (score_matrix[i-1, j-1] +
                self.S[self.b2i(self.seq1[i-1]), self.b2i(self.seq2[j-1])])
        left = score_matrix[i, j-1] + self.G_w()
        up = score_matrix[i-1, j] + self.G_w()
        labels = ['l', 'd', 'u']
        scores = [left, diag, up]
        max_index = np.argmax(scores)
        return scores[max_index], labels[max_index]
    
    def align(self):
        """
        Perform algorithm for global alignment.
           -Score matrix contains the alignment similarity scores
           -Trace matrix contains traceback info from score matrix
        """
        score_matrix = np.empty((len(self.seq1)+1, len(self.seq2)+1), dtype=int)
        score_matrix = self.initialize_score_matrix(score_matrix)
        trace_matrix = np.empty((len(self.seq1)+1, len(self.seq2)+1), dtype='S1')
        trace_matrix[0, :] = 'e'
        trace_matrix[:, 0] = 'e'

        for i in np.arange(1, score_matrix.shape[0]):
            for j in np.arange(1, score_matrix.shape[1]):
                score_ij, label_ij = self.fill_score_ij(score_matrix, i, j)
                score_matrix[i, j] = score_ij
                trace_matrix[i, j] = label_ij
        
        return score_matrix, trace_matrix

    def optimal_alignment(self):
        """
        Use the traceback matrix to build alignment sequence
        """
        seq1 = '-' + self.seq1
        seq2 = '-' + self.seq2
        a1 = ''
        a2 = ''
        i = self.trace.shape[0]-1
        j = self.trace.shape[1]-1
        
        while (i > 0) | (j > 0):
            state = self.trace[i, j]
            if state == 'd':
                a1 += seq1[i]
                a2 += seq2[j]
                i -= 1
                j -= 1
            elif state == 'l':
                a1 += '-'
                a2 += seq2[j]
                j -= 1
            elif state == 'u':
                a1 += seq1[i]
                a2 += '-'
                i -= 1
            elif state == 'e':
                if (i == 0) & (j == 0):
                    a1 += seq1[1]
                    a2 += seq2[1]
                elif (i == 0):
                    a1 += '-'
                    a2 += seq2[j]
                    j -= 1
                else:
                    a1 += seq1[i]
                    a2 += '-'
                    i -= 1
                
        return a1[::-1], a2[::-1]
