from numtheory import mod_exp, crt_inductive
from random import randint
from problem57 import find_residues, brute_force_digest, encrypt_digest

p = 11470374874925275658116663507232161402086650258453896274534991676898999262641581519101074740642369848233294239851519212341844337347119899874391456329785623
q = 335062023296420808191071248367701059461
j = 34233586850807404623475048381328686211071196701374230492615844865929237417097514638999377942356150481334217896204702
g = 622952335333961296978159266084741085889881358738459939978290179936063635566740258555167783009058567397963466103140082647486611657350811560630587013183357

# 0 -> 2^20
# Determined: g^705485 = y1
y1 = 7760073848032689505395005705677365876654629189298052775754597607446617558600394076764814236081991643094239886772481052254010323780165093955236429914607119
# 0 -> 2^40
# Determined: g^359579674340 = y2
y2 = 9388897478013399550694114614498790691034187453089355259602614074132918843899833277397448144245883225611726912025846772975325932794909655215329941809013733

###########################
# Begin Kangaroo parameters
###########################
k = 20
def pseudorandom_map(n):
    return 2 ** (n % k)
N = 0
for i in range(0, k):
    N += pseudorandom_map(i)
N = 4 * (N // k)
print('N', N)

def tame_kangaroo(f, N, g, a, b, y, precomputed_powers):
    # "Tame Kangaroo"
    xT = 0
    yT = mod_exp(g, b, p)
    for _ in range(0, N):
        xT += f(yT)
        yT = (yT * precomputed_powers[f(yT)]) % p

    assert yT == mod_exp(g, b + xT, p), 'Tame kangaroo did not have expected value'

    return xT, yT

def kangaroo_attack(f, N, g, a, b, y):
    # first build a table of g^f(yW) since there's only a finite number of these
    precomputed_powers = {f(x): mod_exp(g, f(x), p) for x in range(0, k)}

    # f is a psuedorandom mapping function
    # g is a generator of the cyclic group
    # b is the upper bound on the discrete logarithm range

    xT, yT = tame_kangaroo(f, N, g, a, b, y, precomputed_powers)
    print('Tame Kangaroo completed', yT)

    # "Wild Kangaroo"
    xW = 0
    yW = y
    iterations = 0
    while xW < b - a + xT:
        iterations += 1
        f_yW = f(yW)
        xW += f_yW
        yW = (yW * precomputed_powers[f_yW]) % p

        if yW == yT:
            # Boom
            assert mod_exp(g, b + xT - xW, p) == y, 'Discrete logarithm was not solved'
            print(f'Finished in {iterations} iterations')
            return b + xT - xW

    raise ValueError('Kangaroo sequences did not intersect')


# g^705485 = y1
# print(kangaroo_attack(pseudorandom_map, N, g, 0, 2**20, y1))
assert mod_exp(g, 705485, p) == y1
# g^359579674340 = y2
# print(kangaroo_attack(pseudorandom_map, N, g, 0, 2**40, y2))
assert mod_exp(g, 359579674340, p) == y2

if __name__ == '__main__':
    bob_secret = randint(0, q)
    residues = list(find_residues(j, p, bob_secret))
    n, r = crt_inductive(residues)
    print(f'secret_key = {n} mod {r}')
    assert bob_secret % r == n, 'Bob secret did not satisfy expected relationship'

    # We know {secret key} = n mod r
    # So {secret key} = n + m * r - must find m
    # Through a series of algebraic transformations, we have:
    # y' = g^{m * r}
    # y' = (g^{r})^m
    g_ = mod_exp(g, r, p)
    # This is the Diffie-Hellman public key (and so public)
    y = mod_exp(g, bob_secret, p)
    g_inverse = mod_exp(g, p - 2, p)
    assert (g * g_inverse) % p == 1, 'g_inverse was not inverse of g'

    y_ = (y * mod_exp(g_inverse, n, p)) % p
    m = kangaroo_attack(pseudorandom_map, N, g_, 0, (q - 1) // r, y_)

    assert n + m * r == bob_secret
    print(f'All done!  Successfully cracked bob_secret using {len(residues)} residues and kangaroo attack')
