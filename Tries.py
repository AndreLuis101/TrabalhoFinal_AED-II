import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import time
import sys

# ==========================================
# PARTE 1: LEITURA DO ARQUIVO FASTA E EXTRAÇÃO
# ==========================================

def read_fasta(file_path):
    """Lê o genoma de referência a partir de um arquivo FASTA."""
    record = SeqIO.read(file_path, "fasta")
    return str(record.seq)

# genome_seq = read_fasta("reference-NC_045512.fasta")
# Para facilitar o teste direto, deixamos uma string genômica simulada.
# Substitua a linha abaixo pela leitura real no ambiente final.
genome_seq = "ATTAAAGGTTTATACCTTCCCAGGTAACAAACCAACCAACTTTCGATCTCTTGTAGATCTGTTCTCTAAA" * 200

# ==========================================
# PARTE 2: ÁRVORE DIGITAL M-ÁRIA (m=4)
# ==========================================

# Mapeamento do alfabeto genético
char_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

class TrieNodeM4:
    def __init__(self):
        # Aloca estritamente 4 ponteiros, caracterizando a árvore m-ária (m=4)
        self.children = [None] * 4 
        self.is_end = False

class TrieM4:
    def __init__(self):
        self.root = TrieNodeM4()
        self.node_count = 1 # Variável para análise de espaço prático

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in char_to_index:
                continue # Ignora caracteres inválidos (ex: 'N' no sequenciamento)
            idx = char_to_index[char]
            if not node.children[idx]:
                node.children[idx] = TrieNodeM4()
                self.node_count += 1
            node = node.children[idx]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in char_to_index:
                return False
            idx = char_to_index[char]
            if not node.children[idx]:
                return False
            node = node.children[idx]
        return node.is_end

# ==========================================
# PARTE 3: ÁRVORE DIGITAL BINÁRIA (m=2)
# ==========================================

# Mapeamento de nucleotídeos para 2 bits
dna_to_bin = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}

class TrieNodeBin:
    def __init__(self):
        # Aloca estritamente 2 ponteiros (0 ou 1)
        self.children = [None] * 2
        self.is_end = False

class TrieBin:
    def __init__(self):
        self.root = TrieNodeBin()
        self.node_count = 1 # Para análise de espaço prático

    def insert(self, word):
        node = self.root
        # Conversão prévia da sequência de letras para string de bits
        bin_word = "".join([dna_to_bin[c] for c in word if c in dna_to_bin])
        
        for bit in bin_word:
            idx = int(bit)
            if not node.children[idx]:
                node.children[idx] = TrieNodeBin()
                self.node_count += 1
            node = node.children[idx]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        bin_word = "".join([dna_to_bin[c] for c in word if c in dna_to_bin])
        for bit in bin_word:
            idx = int(bit)
            if not node.children[idx]:
                return False
            node = node.children[idx]
        return node.is_end

# ==========================================
# PARTE 4: TESTES DE COMPARAÇÃO (TEMPO E ESPAÇO)
# ==========================================

slice_seq = genome_seq[:10000] # Pegando os primeiros 10 mil nucleotídeos para teste
k = 21 # Tamanho típico de um k-mer para buscas genômicas
kmers = [slice_seq[i:i+k] for i in range(len(slice_seq)-k+1)]
conjunto_kmers = list(set(kmers)) # Removemos duplicatas para inserção

print(f"Total de sequências únicas a inserir: {len(conjunto_kmers)}\n")

# Simulando o padrão de uma variante do vírus (buscando um k-mer específico)
# Pegamos um padrão que sabemos que existe no meio do nosso conjunto
padrao_variante = conjunto_kmers[len(conjunto_kmers) // 2]
numero_de_buscas = 10000 # Executamos a busca várias vezes para o tempo ser mensurável no time.time()

# --- TESTE DA ÁRVORE M-ÁRIA (m=4) ---
trie_m4 = TrieM4()

# Medição de Inserção M-ária
start_time_m4 = time.time()
for kmer in conjunto_kmers:
    trie_m4.insert(kmer)
end_time_m4 = time.time()

tempo_m4 = end_time_m4 - start_time_m4
nos_m4 = trie_m4.node_count
tamanho_memoria_m4 = nos_m4 * sys.getsizeof(TrieNodeM4())

# Medição de Busca M-ária
start_search_m4 = time.time()
for _ in range(numero_de_buscas):
    encontrado_m4 = trie_m4.search(padrao_variante)
end_search_m4 = time.time()
tempo_busca_m4 = (end_search_m4 - start_search_m4) / numero_de_buscas

print("--- RESULTADOS ÁRVORE m-ária (m=4) ---")
print(f"Tempo total de Inserção: {tempo_m4:.6f} segundos")
print(f"Nós criados: {nos_m4}")
print(f"Estimativa de Memória (bytes): {tamanho_memoria_m4}")
print(f"Busca pelo padrão '{padrao_variante}': {'Encontrado' if encontrado_m4 else 'Não encontrado'}")
print(f"Tempo médio de Busca (por padrão): {tempo_busca_m4:.8f} segundos\n")


# --- TESTE DA ÁRVORE BINÁRIA ---
trie_bin = TrieBin()

# Medição de Inserção Binária
start_time_bin = time.time()
for kmer in conjunto_kmers:
    trie_bin.insert(kmer)
end_time_bin = time.time()

tempo_bin = end_time_bin - start_time_bin
nos_bin = trie_bin.node_count
tamanho_memoria_bin = nos_bin * sys.getsizeof(TrieNodeBin())

# Medição de Busca Binária
start_search_bin = time.time()
for _ in range(numero_de_buscas):
    encontrado_bin = trie_bin.search(padrao_variante)
end_search_bin = time.time()
tempo_busca_bin = (end_search_bin - start_search_bin) / numero_de_buscas

print("--- RESULTADOS ÁRVORE BINÁRIA ---")
print(f"Tempo total de Inserção: {tempo_bin:.6f} segundos")
print(f"Nós criados: {nos_bin}")
print(f"Estimativa de Memória (bytes): {tamanho_memoria_bin}")
print(f"Busca pelo padrão '{padrao_variante}': {'Encontrado' if encontrado_bin else 'Não encontrado'}")
print(f"Tempo médio de Busca (por padrão): {tempo_busca_bin:.8f} segundos\n")

# ==========================================
# FUNÇÕES DE POSICIONAMENTO HIERÁRQUICO (ARVORESCÊNCIA)
# ==========================================
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Calcula as coordenadas x, y para dispor o grafo em níveis hierárquicos."""
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.successors(root))
    if not isinstance(children, list):
        children = list(children)
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                 vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root)
    return pos

# ==========================================
# PARÂMETROS E DADOS DE EXEMPLO
# ==========================================
# Utilizando um conjunto pequeno de sequências (k-mers) para que a árvore fique legível na imagem
kmers_exemplo = ["ATTAG", "CATGA", "TCGAT", "ATGAT"]
# Mapeamento para a Árvore Binária (2 bits por nucleotídeo)
dna_to_bin = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}

# ==========================================
# 1. CONSTRUÇÃO DO GRAFO M-ÁRIO (m=4)
# ==========================================
G_m4 = nx.DiGraph()
G_m4.add_node("root", label="Raiz")

for kmer in kmers_exemplo:
    current_node = "root"
    for char in kmer:
        # Cria um identificador único para o nó no caminho
        next_node = f"{current_node}_{char}"
        if not G_m4.has_node(next_node):
            G_m4.add_node(next_node, label=char)
            G_m4.add_edge(current_node, next_node)
        current_node = next_node

# ==========================================
# 2. CONSTRUÇÃO DO GRAFO BINÁRIO (m=2)
# ==========================================
G_bin = nx.DiGraph()
G_bin.add_node("root", label="Raiz")

for kmer in kmers_exemplo:
    # Converte o k-mer biológico para uma string de bits
    bin_kmer = "".join([dna_to_bin[c] for c in kmer])
    
    current_node = "root"
    for bit in bin_kmer:
        # Cria um identificador único para o nó no caminho binário
        next_node = f"{current_node}_{bit}"
        if not G_bin.has_node(next_node):
            G_bin.add_node(next_node, label=bit)
            G_bin.add_edge(current_node, next_node)
        current_node = next_node

# ==========================================
# 3. PLOTAGEM COMPARATIVA (LADO A LADO)
# ==========================================
plt.figure(figsize=(16, 8))

# Subplot 1: Árvore m-ária
plt.subplot(1, 2, 1)
pos_m4 = hierarchy_pos(G_m4, root="root", width=2.0, vert_gap=0.3)
labels_m4 = nx.get_node_attributes(G_m4, 'label')
nx.draw(G_m4, pos_m4, with_labels=True, labels=labels_m4, node_size=900, 
        node_color="lightblue", font_size=12, font_weight="bold", arrows=True)
plt.title("Árvore Digital m-ária (m=4)\nAlfabeto: {A, C, G, T}", fontsize=14)

# Subplot 2: Árvore Binária
plt.subplot(1, 2, 2)
pos_bin = hierarchy_pos(G_bin, root="root", width=2.0, vert_gap=0.3)
labels_bin = nx.get_node_attributes(G_bin, 'label')
nx.draw(G_bin, pos_bin, with_labels=True, labels=labels_bin, node_size=900, 
        node_color="lightgreen", font_size=12, font_weight="bold", arrows=True)
plt.title("Árvore Digital Binária (m=2)\nAlfabeto: {0, 1}", fontsize=14)

# Título Principal
plt.suptitle(f"Comparação Visual das Árvores para as sequências: {kmers_exemplo}", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ==========================================
# 4. CONSTRUÇÃO E PLOTAGEM DAS ÁRVORES COMPLETAS (SEPARADAS)
# ==========================================
# AVISO: Estamos extraindo uma amostra do arquivo real lido (conjunto_kmers)
# para evitar o travamento de memória (Out Of Memory) na renderização do Matplotlib,
# uma vez que o genoma real geraria dezenas de milhares de nós no gráfico.

limitador_plot = 30 # Aumente este número gradativamente conforme a capacidade do seu computador
kmers_arquivo_real = list(conjunto_kmers)[:limitador_plot] 

print(f"Gerando gráficos finais isolados para {limitador_plot} k-mers reais do genoma...")

# ---------------------------------------------------------
# 4.1. GRAFO FINAL: ÁRVORE M-ÁRIA (m=4) DO ARQUIVO REAL
# ---------------------------------------------------------
G_m4_final = nx.DiGraph()
G_m4_final.add_node("root", label="Raiz")

for kmer in kmers_arquivo_real:
    current_node = "root"
    for char in kmer:
        next_node = f"{current_node}_{char}"
        if not G_m4_final.has_node(next_node):
            G_m4_final.add_node(next_node, label=char)
            G_m4_final.add_edge(current_node, next_node)
        current_node = next_node

plt.figure(figsize=(24, 12)) # Tamanho maior para acomodar a árvore expandida
pos_m4_final = hierarchy_pos(G_m4_final, root="root", width=2.0, vert_gap=0.2)
labels_m4_final = nx.get_node_attributes(G_m4_final, 'label')

nx.draw(G_m4_final, pos_m4_final, with_labels=True, labels=labels_m4_final, 
        node_size=400, node_color="lightblue", font_size=9, font_weight="bold", 
        arrows=False, edge_color="gray")

plt.title("Representação Final Separada: Árvore Digital m-ária (m=4)\nAmostra do Genoma de Referência do SARS-CoV-2", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4.2. GRAFO FINAL: ÁRVORE BINÁRIA (m=2) DO ARQUIVO REAL
# ---------------------------------------------------------
G_bin_final = nx.DiGraph()
G_bin_final.add_node("root", label="Raiz")

for kmer in kmers_arquivo_real:
    # Conversão do trecho real para bits
    bin_kmer = "".join([dna_to_bin.get(c, "") for c in kmer])
    
    current_node = "root"
    for bit in bin_kmer:
        next_node = f"{current_node}_{bit}"
        if not G_bin_final.has_node(next_node):
            G_bin_final.add_node(next_node, label=bit)
            G_bin_final.add_edge(current_node, next_node)
        current_node = next_node

plt.figure(figsize=(24, 14)) # Tamanho ainda maior devido à profundidade que dobra (2 bits por nucleotídeo)
pos_bin_final = hierarchy_pos(G_bin_final, root="root", width=3.0, vert_gap=0.15)
labels_bin_final = nx.get_node_attributes(G_bin_final, 'label')

nx.draw(G_bin_final, pos_bin_final, with_labels=True, labels=labels_bin_final, 
        node_size=300, node_color="lightgreen", font_size=8, font_weight="bold", 
        arrows=False, edge_color="gray")

plt.title("Representação Final Separada: Árvore Digital Binária (m=2)\nAmostra do Genoma de Referência do SARS-CoV-2", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.show()