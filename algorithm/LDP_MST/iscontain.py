def iscontain(q,x):
    m = len(q)
    flag = 0
    for i in range(m):
        if q[i] == x:
            flag = 1
            break
    return flag