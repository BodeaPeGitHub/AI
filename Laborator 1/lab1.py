def ultimul_cuvant(propozitie):
	"""
	Algoritmul cauta maximul dintr-o propozitie.
	In: propozitia.
	Out: cuvantul maxim din propozitie
	"""
	maxim = 'a'
	for cuvant in propozitie.split(" "):
		if maxim < cuvant:
			maxim = cuvant
	return maxim


def test_ultimul_cuvant():
	assert(ultimul_cuvant("Ana are mere rosii si galbene") == "si")
	assert(ultimul_cuvant("Ana are mere rosii si galbene dar ii place zucchini") == "zucchini")


def distanta_euclidiana(puncte):
	'''
	Algoritm care calculeaza distanta euclidiana dintre doua puncte.
	In: un tuple care are 2 tupluri, reprezentand punctele
	Out: distanta euclidiana (double)
	'''
	import math
	punctA = puncte[0]
	punctB = puncte[1]
	rez = 0
	for i in range(0, len(punctA)):
		rez += (punctB[i] - punctA[i]) ** 2
	return math.sqrt(rez)
	

def test_distanta_euclidiana():
	lmb = 0.0001
	assert(distanta_euclidiana(((1, 5), (4, 1))) - 5 < lmb)
	assert(distanta_euclidiana(((1, 2), (3, 4))) - 2.8284 < lmb)

def produs_scalar_vectori_rari(vector1, vector2):
	"""
	Problema 3.
	Functie care calculeaza produsul scalar pentru doi vectori.
	In: cei doi vectori ca liste.
	Out: produsul sclar a celor doi vectori. (double)	
	"""
	produs_scalar = 0
	for elem1, elem2 in zip(vector1, vector2):
		produs_scalar = produs_scalar + elem1 * elem2
	return produs_scalar;


def test_produs_scalar_vectori_rari():
	lbd = 0.001
	assert(produs_scalar_vectori_rari([1, 0, 2, 0, 3], [1, 2, 0, 3, 1]) - 4 < lbd)
	assert(produs_scalar_vectori_rari([2, 0, 2, 0, 1], [0, 1, 0, 7, 0]) - 0 < lbd)
	assert(produs_scalar_vectori_rari([2], [-4]) - -8 < lbd)
	assert(produs_scalar_vectori_rari([0.1, 0.2, 0], [0.3, 0, 0.4]) - 0.03 < lbd)


def cuvinte_care_apar_o_singura_data(propozitie):
	"""
	Problema 4.
	Functie care verifica ce cuvinte apar o singura data in propozitie. 
	In: propozitia (string)
	Out: cuvintele care apar o singura data (lista de string-uri)	
	"""
	frecventa = {}
	for cuvant in propozitie.split():
		if cuvant in frecventa:
			frecventa[cuvant] = False
		else:
			frecventa[cuvant] = True
	return [cuvant for cuvant in frecventa if frecventa[cuvant] == True] 


def test_cuvinte_care_apar_o_singura_data():
	assert(cuvinte_care_apar_o_singura_data("ana are ana are mere rosii ana") == ["mere", "rosii"])
	assert(cuvinte_care_apar_o_singura_data("ana are mere") == ["ana", "are", "mere"])
	assert(cuvinte_care_apar_o_singura_data("") == [])
	

def identificare_valoare_care_se_repeta(sir):
	"""
	Problema 5.
	Functie care cauta numarul care se repeta dintr-o lista de n numere cu valori de la 1 - n.
	In: lista data.
	Out: numarul care se repeta.
	"""
	suma_posibila_numerelor = (len(sir) * (len(sir) - 1)) / 2 
	suma_numerelor = sum(sir)
	return suma_numerelor - suma_posibila_numerelor
	
def test_identificare_valoare_care_se_repeta():
	assert(identificare_valoare_care_se_repeta([1, 2, 3, 4, 2]) == 2)		
	assert(identificare_valoare_care_se_repeta([1, 2, 3, 4, 1]) == 1)		
	assert(identificare_valoare_care_se_repeta([2, 2, 1]) == 2)		


def cautare_numar_majoritar(sir):
	"""
	Problema 6.
	Functie care cauta elementul care apare de mai multe ori decat n/2, unde n - lungimea sirului.
	In: sirul dat. (lista)
	Out: elementul majoritar. 
	Tip: Boyer Moore Vote Algorithm
	"""
	candidat = sir[0]
	contor = 0
	for elem in sir:
		if candidat == elem:
			contor += 1
		else:
			contor = 1
			candidat = elem
	if contor > len(sir) // 2:
		return candidat
	contor = 0
	i = 0
	while i < len(sir) and contor < len(sir) // 2:
		if candidat == sir[i]:
			contor += 1
		i += 1
	return candidat	


def test_cautare_numar_majoritar():
	assert(cautare_numar_majoritar([2, 8, 7, 2, 2, 5, 2, 3, 1, 2, 2]) == 2) 
	assert(cautare_numar_majoritar(['A', 'B' 'A', 'C', 'A', 'B', 'A']) == 'A')


def al_k_lea_cel_mai_mare_element(sir, k):
	"""
	Problema 7.
	Functie care cauta al k-lea cel mai mare element dintr-un sir.
	In: sirul (nevid) si k-ul (k <= ca lungimea sirului)
	Out: elementul cautat
	Tip: se foloseste un minheap (pentru ca python nu are maxheap)
	"""
	import heapq
	sir = [-1 * elem for elem in sir]
	heapq.heapify(sir)
	while(k != 1):
		heapq.heappop(sir)
		k -= 1
	return -heapq.heappop(sir)	


def test_al_k_lea_cel_mai_mare_element():
	assert(al_k_lea_cel_mai_mare_element([7,4,6,3,9,1], 2) == 7)
	assert(al_k_lea_cel_mai_mare_element([7,4,6,3,9,1], 1) == 9)
	assert(al_k_lea_cel_mai_mare_element([7,4,6,3,9,1], 3) == 6)
	assert(al_k_lea_cel_mai_mare_element([7,4,6,3,9,1], 6) == 1)
	assert(al_k_lea_cel_mai_mare_element([7], 1) == 7)


def formeaza_binare_de_la_1_la_n(numar):
	"""
	Problema 8.
	Functie care returneaza o lista cu numerele bineare de la 1 la n.
	In: numarul n 
	Out: lista de striguri care reprezinta numerele binare de la 1 la n.
	Tip: se foloseste o coada in care se adauga pe rand string-uri de forma str + 0, str + 1
	"""
	from queue import Queue	
	queue = Queue()
	queue.put(str(1))
	binare = []
	for iteratii in range(numar):
		temp = queue.get()
		binare.append(temp)
		queue.put(temp + '0')		
		queue.put(temp + '1')		
	return binare


def test_formeaza_binare_de_la_1_la_n():
	assert(formeaza_binare_de_la_1_la_n(2) == ['1', '10'])
	assert(formeaza_binare_de_la_1_la_n(4) == ['1', '10', '11', '100'])
	assert(formeaza_binare_de_la_1_la_n(1) == ['1'])
	

def calculare_suma_submatrice(matrix, puncte):
	"""
	Functie care calculeaza suma din matricea sumelor si un punct.
	Returneaza None daca primul punct este mai mare decat cel de al 2-lea.
	In: matricea sumelor, perechea de puncte.
	Out: suma sau None 
	"""
	if puncte[0] > puncte[1]:
		return None
	if puncte[0] == (0, 0):
		return matrix[puncte[1][0]][puncte[1][1]]
	A = (puncte[0][0] - 1, puncte[0][1] - 1)
	B = puncte[1]
	if A[0] < 0:
		return matrix[B[0]][B[1]] - matrix[B[0]][A[1]]	
	if A[1] < 0:
		return matrix[B[0]][B[1]] - matrix[A[0]][B[1]]	
	return matrix[B[0]][B[1]] + matrix[A[0]][A[1]] - matrix[A[0]][B[1]] - matrix[B[0]][A[1]]


def	suma_sub_matricilor(matrix, pereche1, pereche2):
	"""
	Problema 9.
	Functia returneaza suma submatriciilor definite de cele doua puncte.
	Intr-o pereche trebuie sa fie primul punct mai mic sau egal cu cel de al 2-lea.
	In: matricea initiala (lista de liste) si cele 2 perechi de puncte (tuple de tuple-uri).
	Out: un tuple cu cele 2 sume sau none in cazul in care sunt date date gresite	
	"""	
	for index in range(1, len(matrix[0])):
		matrix[0][index] += matrix[0][index - 1]	
	for index in range(1, len(matrix)):
		matrix[index][0] += matrix[index - 1][0] 
	for i in range(1, len(matrix)):
		for j in range(1, len(matrix[i])):
			matrix[i][j] += matrix[i - 1][j] + matrix[i][j - 1]	- matrix[i - 1][j - 1]	
	return calculare_suma_submatrice(matrix, pereche1), calculare_suma_submatrice(matrix, pereche2)


def test_suma_sub_matricilor():
	assert(suma_sub_matricilor([
						[0, 2, 5, 4, 1],
						[4, 8, 2, 3, 7],
						[6, 3, 4, 6, 2],
						[7, 3, 1, 8, 3],
						[1, 5, 7, 9, 4]], ((1, 1), (3, 3)), ((2, 2), (4, 4))) == (38, 44)) 
	assert(suma_sub_matricilor([
						[0, 2, 5, 4, 1],
						[4, 8, 2, 3, 7],
						[6, 3, 4, 6, 2],
						[7, 3, 1, 8, 3],
						[1, 5, 7, 9, 4]], ((5, 5), (3, 3)), ((2, 2), (4, 4))) == (None, 44))

	assert(suma_sub_matricilor([
						[0, 2, 5, 4, 1],
						[4, 8, 2, 3, 7],
						[6, 3, 4, 6, 2],
						[7, 3, 1, 8, 3],
						[1, 5, 7, 9, 4]], ((1, 1), (3, 3)), ((5, 5), (4, 4))) == (38, None))
	assert(suma_sub_matricilor([
						[0, 2, 5, 4, 1],
						[4, 8, 2, 3, 7],
						[6, 3, 4, 6, 2],
						[7, 3, 1, 8, 3],
						[1, 5, 7, 9, 4]], ((0, 1), (2, 2)), ((1, 0), (2, 2))) == (24, 27))
	assert(suma_sub_matricilor([
						[0, 2, 5, 4, 1],
						[4, 8, 2, 3, 7],
						[6, 3, 4, 6, 2],
						[7, 3, 1, 8, 3],
						[1, 5, 7, 9, 4]], ((0, 0), (1, 1)), ((1, 1), (1, 1))) == (14, 8))
	

def linia_cu_cele_mai_multe_elemente_de_1(matrix):
	"""
	Functie care cauta care este randul cu numarul maxim de 1-uri.
	In: matricea
	Out: indexul randului care are numarul maxim de 1-uri.
	Se face o cautare binara pentru fiecare rand care returneaza prima pozitie unde se gaseste un 1. 
	Dupa care intr-un vector in care se pun toate aceste valori se cauta valoarea minima iar indexul la care se gaseste este cel cautat de noi.
	"""
	def cautare_binara(vector, inc, fin):
		if vector[0] == 1:
			return -1
		if inc > fin:
			return len(vector) + 1
		mid = (inc + fin) // 2	
		if vector[mid : mid + 2] == [0, 1]:
			return mid + 1 
		if vector[mid - 1 : mid + 1] == [0, 1]:
			return mid
		if vector[mid] == 1:
			return cautare_binara(vector, inc, mid - 1)
		return cautare_binara(vector, mid + 1, fin)
	
	def caut_maxim():
		maxims = [cautare_binara(row, 0, len(row) - 1) for row in matrix]
		return maxims.index(min(maxims))
				
	return caut_maxim()
		

def test_linia_cu_cele_mai_multe_elemente_de_1():
	assert(linia_cu_cele_mai_multe_elemente_de_1([
											[1,1,1,1,1],
											[0,0,0,0,0],
											[0,1,1,1,1],
											[0,0,1,1,1]
											]) == 0)
	assert(linia_cu_cele_mai_multe_elemente_de_1([
											[0,0,0,0,0],
											[0,0,0,0,0],
											[0,0,0,0,0],
											]) == 0)
	assert(linia_cu_cele_mai_multe_elemente_de_1([
											[0,0,1,1,1],
											[0,0,0,0,0],
											[0,1,1,1,1],
											[0,0,1,1,1]
											]) == 2)
	assert(linia_cu_cele_mai_multe_elemente_de_1([
											[0,1,1,1,1],
											[0,0,0,0,0],
											[0,1,1,1,1],
											[1,1,1,1,1]
											]) == 3)
	assert(linia_cu_cele_mai_multe_elemente_de_1([
											[0, 0, 0, 0, 0],
          					           		[0, 0, 0, 0, 0],
                      						[0, 0, 0, 1, 1],
                      						[0, 0, 0, 0, 1],
                      						[0, 0, 0, 0, 0],
                      						[0, 1, 1, 1, 1],
                      						[1, 1, 1, 1, 1]
											]) == 6)
	

def inconjurate_complet(matrix):
	'''
	Functie care inlocuieste toate valorile de 0 care sunt complet inconjurate de 1 cu valori de 1
	IN: matricea cu valorile date
	Out: matricea dupa inlocuire
	Se parcurge rama matricei iar in momentul cand se gaseste un element care este egal cu 0 se apeleaza algoritmul lui lee care cauta toti vecinii lui care sunt egali cu 0.
	Toate elementele gasite se pun intr-o matrice care are initial doar elemente egale cu 1.
	Acea matrice se returneaza la final.
	'''
	poz_x = [-1,  0, 0, 1]
	poz_y = [ 0, -1, 1, 0]
	final_matrix = [[1 for elem in matrix[0]] for elem in matrix] 
	
	def lee(rand, coloana):
		'''
		Algoritmul lui Lee, cauta toti vecinii unui element care sunt egali cu 0 si cand ii gaseste pune 0 in matricea finala pe aceiasi pozitie.
		In: randul si coloana elementului.
		Out: nu returneaza nimic, dar modificarile se observa in matricea finala. 
		'''
		if rand in range(len(matrix)) and coloana in range(len(matrix[0])) and final_matrix[rand][coloana] != 0 and matrix[rand][coloana] == 0:
			final_matrix[rand][coloana] = 0
			for x, y in zip(poz_x, poz_y):
				lee(rand + x, coloana + y)

	def parcurgere_rama():
		'''
		Algoritmul care parcurge rama matricei si apeleaza lee pentru fiecare element.
		IN: -
		Out: - 
		Modificarile se observa in matricea finala.
		'''
		end_matrix = len(matrix) - 1
		for j in range(len(matrix[0])):
			lee(0, j)	
			lee(end_matrix, j)
	
		end_col = len(matrix[0]) - 1
		for i in range(len(matrix)):
			lee(i, 0)
			lee(i, end_col)
			
	parcurgere_rama()
	return final_matrix


def test_inconjurate_complet():
	assert(inconjurate_complet([
								[1,1,1,1,0,0,1,1,0,1],
								[1,0,0,1,1,0,1,1,1,1],
								[1,0,0,1,1,1,1,1,1,1],
								[1,1,1,1,0,0,1,1,0,1],
								[1,0,0,1,1,0,1,1,0,0],
								[1,1,0,1,1,0,0,1,0,1],
								[1,1,1,0,1,0,1,0,0,1],
								[1,1,1,0,1,1,1,1,1,1]
								]) 
								== 
								[
								[1,1,1,1,0,0,1,1,0,1],
								[1,1,1,1,1,0,1,1,1,1],
								[1,1,1,1,1,1,1,1,1,1],
								[1,1,1,1,1,1,1,1,0,1],
								[1,1,1,1,1,1,1,1,0,0],
								[1,1,1,1,1,1,1,1,0,1],
								[1,1,1,0,1,1,1,0,0,1],
								[1,1,1,0,1,1,1,1,1,1]
								]) 
	assert(inconjurate_complet([
								[1,1,1,1,0,0,1,1,0,1],
								[1,0,0,1,1,0,1,1,1,1],
								[1,0,0,1,1,1,1,1,1,1],
								[1,1,1,1,0,0,1,1,0,1],
								[0,0,0,1,1,0,1,1,0,0],
								[1,1,0,1,1,0,0,1,0,1],
								[1,1,1,0,1,0,0,0,0,1],
								[1,1,1,0,1,1,1,1,1,1]
								]) 
								== 
								[
								[1,1,1,1,0,0,1,1,0,1],
								[1,1,1,1,1,0,1,1,1,1],
								[1,1,1,1,1,1,1,1,1,1],
								[1,1,1,1,0,0,1,1,0,1],
								[0,0,0,1,1,0,1,1,0,0],
								[1,1,0,1,1,0,0,1,0,1],
								[1,1,1,0,1,0,0,0,0,1],
								[1,1,1,0,1,1,1,1,1,1]
								]) 




if __name__ == "__main__":
	test_ultimul_cuvant() #1
	test_distanta_euclidiana() #2
	test_produs_scalar_vectori_rari() #3
	test_cuvinte_care_apar_o_singura_data() #4
	test_identificare_valoare_care_se_repeta() #5
	test_cautare_numar_majoritar() #6
	test_al_k_lea_cel_mai_mare_element() #7
	test_formeaza_binare_de_la_1_la_n() #8
	test_suma_sub_matricilor() #9
	test_linia_cu_cele_mai_multe_elemente_de_1() #10
	test_inconjurate_complet() #11
	print('Totul merge ca pe roate')

