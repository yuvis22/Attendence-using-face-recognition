#Function
def union1 (a1, a2, a3) :
    setl = set (al)
    set2 = set (a2)
    set3 = set (a3)
    setl = setl.intersection (set2)
    answer = setl.union (set3)
    print(list(answer))

n=111
al = []
a2 =[]
a3 = []
for i in range (n):
    if i%7==0:
        al.append (i)
    if i%5==0:
        a2.append (i)
    if i%3==0:
        a3.append (i)
union1 (al, a2, a3)