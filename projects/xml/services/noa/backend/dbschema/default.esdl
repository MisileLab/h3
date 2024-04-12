module default {
  type KeyStore {
    name: str;
    pubkey: str;
    privkey: str;
  }

  type User {
    multi groups: KeyStore {
      name: str;
      pubkey: str;
    }
  }
}
