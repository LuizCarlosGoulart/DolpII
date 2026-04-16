from src.core import default_field_dictionary_path, load_field_dictionary


def test_field_dictionary_loads_required_categories_and_core_fields() -> None:
    dictionary = load_field_dictionary()
    fields = dictionary.by_internal_name()
    categories = {field.category for field in dictionary.fields}

    assert default_field_dictionary_path().name == "field_dictionary.yaml"
    assert dictionary.version == "1.0"
    assert categories == {
        "note_identification",
        "provider",
        "recipient",
        "service",
        "taxation_values",
        "discounts_retentions",
        "net_amount",
    }
    assert "nfse_number" in fields
    assert "provider_document" in fields
    assert "recipient_name" in fields
    assert "service_description" in fields
    assert "gross_amount" in fields
    assert "iss_withheld_amount" in fields
    assert "net_amount" in fields


def test_field_dictionary_builds_alias_index_for_canonical_lookup() -> None:
    dictionary = load_field_dictionary()
    alias_index = dictionary.alias_index()

    assert alias_index["nfse_number"] == "nfse_number"
    assert alias_index["numero_nfse"] == "nfse_number"
    assert alias_index["cnpj_prestador"] == "provider_document"
    assert alias_index["descricao_servico"] == "service_description"
    assert alias_index["valor_liquido"] == "net_amount"
