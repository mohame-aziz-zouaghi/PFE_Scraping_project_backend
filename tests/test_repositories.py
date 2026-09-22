from app.repositories.product_repo import ProductRepository


def test_product_sort_fields_are_allowlisted():
    assert "price" in ProductRepository.SORT_FIELDS
    assert "__class__" not in ProductRepository.SORT_FIELDS
