def process_two_arrays(arr1: list, arr2: list, f1, f2):
    n = len(arr1)
    m = len(arr2)
    k = min([n, m])

    for i in range(k):
        f1(arr1[i])
        f2(arr2[i])
    for i in range(k, n):
        f1(arr1[i])
    for i in range(k, m):
        f2(arr2[i])


def check_if_duplicates(arr1: list, arr2: list):
    duplicates = []
    
    def f1(obj1):
        if (obj1 in arr2) and (obj1 not in duplicates):
            duplicates.append(obj1)

    def f2(obj2):
        if (obj2 in arr1) and (obj2 not in duplicates):
            duplicates.append(obj2)

    process_two_arrays(arr1, arr2, f1, f2)

    return len(duplicates) == 0
