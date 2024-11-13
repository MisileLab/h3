module theresa {
  type Letter {
    required name: str {constraint exclusive};
    required tldr: str;
    multi signers: User;
  }

  type User {
    required name: str;
    required email: str {constraint exclusive};
    required message: str;
    required hash: str {constraint exclusive};
    signature: bytes;
  }
}
