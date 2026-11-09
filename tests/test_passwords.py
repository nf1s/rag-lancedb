from common.passwords import hash_password, verify_password


def test_hash_and_verify_roundtrip():
    encoded = hash_password("correct horse battery staple")
    assert encoded.startswith("scrypt$")
    assert verify_password("correct horse battery staple", encoded)


def test_wrong_password_is_rejected():
    encoded = hash_password("secret")
    assert not verify_password("Secret", encoded)
    assert not verify_password("", encoded)


def test_hashes_are_salted():
    first = hash_password("same password")
    second = hash_password("same password")
    assert first != second
    assert verify_password("same password", first)
    assert verify_password("same password", second)


def test_malformed_hash_is_rejected():
    assert not verify_password("secret", "not-a-hash")
    assert not verify_password("secret", "scrypt$1$2$3$bad")
