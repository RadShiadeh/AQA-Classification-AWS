def diffIsOne(nums):
    mapping = {}

    for x in nums:
        d = 1 - x
        if abs(d) in mapping:
            print(mapping)
            return True
        
        mapping[x] = x

    print(mapping)
    return False


n = [5,5,5,5]
n1 = [1,2,3]
print(diffIsOne(n))
print(diffIsOne(n1))