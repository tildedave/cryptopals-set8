from random import randint
from typing import NamedTuple, Tuple


from elliptic_curve import EllipticCurve, EllipticCurvePoint


class ECDHKeypair(NamedTuple):
    curve: EllipticCurve
    secret: int
    public: EllipticCurvePoint

    def compute_secret(self,
                       peer_public: EllipticCurvePoint,
                       ) -> EllipticCurvePoint:
        """
        Compute the public key ^ secret.
        """
        return self.curve.scalar_mult(peer_public, self.secret)


class DiffieHelman:
    def __init__(self,
                 curve: EllipticCurve,
                 point: EllipticCurvePoint,
                 point_order: int,
                 ):
        self.curve = curve
        self.point = point
        self.point_order = point_order

    def generate_keypair(self) -> ECDHKeypair:
        """
        Generate an ECDH keypair.

        Alice and Bob agree to use the given curve and start with a given point on
        a curve.  (Public information)

        Alice and Bob both generate a random integer between 0 and the order of the
        point on the curve. (Secret information)

        Diffie-Hellmen then works as follows:
        Alice keypair: (secret_A, g^secret_A)
        Bob keypair: (secret_B, g^secret_B)

        Alice and Bob then create the shared key:
            (g^secret_A)^(secret_B) = (g^secret_B)^(secret_A)
        """
        secret = randint(0, self.point_order)
        public = self.curve.scalar_mult(self.point, secret)

        return ECDHKeypair(self.curve, secret, public)
