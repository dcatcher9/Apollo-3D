/**
 * @file tests/unit/test_crypto.cpp
 * @brief Test src/crypto.*.
 */
#include "../tests_common.h"

#include <openssl/x509.h>

#include <src/crypto.h>

TEST(CryptoTest, GeneratedCredentialsExposeSubjectAndVerifySignatures) {
  constexpr std::string_view common_name = "Apollo Test Host";
  constexpr std::string_view payload = "payload";

  auto creds = crypto::gen_creds(common_name, 2048);
  ASSERT_FALSE(creds.x509.empty());
  ASSERT_FALSE(creds.pkey.empty());

  auto cert = crypto::x509(creds.x509);
  auto pkey = crypto::pkey(creds.pkey);
  ASSERT_NE(cert.get(), nullptr);
  ASSERT_NE(pkey.get(), nullptr);

  const auto subject = X509_get_subject_name(cert.get());
  ASSERT_NE(subject, nullptr);

  const auto common_name_index = X509_NAME_get_index_by_NID(subject, NID_commonName, -1);
  ASSERT_GE(common_name_index, 0);

  const auto common_name_entry = X509_NAME_get_entry(subject, common_name_index);
  ASSERT_NE(common_name_entry, nullptr);

  const auto common_name_data = X509_NAME_ENTRY_get_data(common_name_entry);
  ASSERT_NE(common_name_data, nullptr);

  const std::string_view parsed_common_name {
    reinterpret_cast<const char *>(ASN1_STRING_get0_data(common_name_data)),
    static_cast<std::size_t>(ASN1_STRING_length(common_name_data))
  };
  ASSERT_EQ(parsed_common_name, common_name);

  const auto issuer = X509_get_issuer_name(cert.get());
  ASSERT_NE(issuer, nullptr);
  ASSERT_EQ(X509_NAME_cmp(subject, issuer), 0);
  ASSERT_EQ(X509_verify(cert.get(), X509_get0_pubkey(cert.get())), 1);

  ASSERT_FALSE(crypto::signature(cert).empty());

  const auto signed_payload = crypto::sign256(pkey, payload);
  ASSERT_FALSE(signed_payload.empty());
  ASSERT_TRUE(crypto::verify256(
    cert,
    payload,
    {reinterpret_cast<const char *>(signed_payload.data()), signed_payload.size()}
  ));
}
