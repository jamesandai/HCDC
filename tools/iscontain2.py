def iscontain2(q,front,rear,x):
    flag = 0
    for i in range(front,rear):
        if q[i] == x:
            flag = 1
            break
    return flag

