"""
Comprehensive matching tests across categories and variant types.

Every test is a small "would a human say these are the same product?"
judgment. The expected `decision` reflects what the matcher SHOULD do,
not what it currently does.

Run:
    pytest tests/test_matching_rules.py -v
"""
from types import SimpleNamespace

import pytest

from app.services.matching_rules import compare_products


def P(name: str, brand=None, category=None, desc=None):
    """Helper to build a Product-like object."""
    return SimpleNamespace(
        name=name,
        brand=brand,
        category=category,
        short_description=desc,
    )


def decide(a, b) -> str:
    return compare_products(a=a, b=b).decision


def reason(a, b) -> str:
    return compare_products(a=a, b=b).reason


def overlap(a, b) -> float:
    return compare_products(a=a, b=b).token_overlap


# ═══════════════════════════════════════════════════════
# CATEGORY 1 — LAPTOPS with CPU generation variants
# ═══════════════════════════════════════════════════════
class TestLaptops:
    def test_same_laptop_different_gen_naming(self):
        """'i5-13450HX' vs 'i5 13ème Gén' — same CPU, same laptop."""
        a = P(
            "Lenovo LOQ 15IRX9 | Intel Core i5-13450HX | 16GB | RTX 4050 | 512GB NVMe",
            brand="Lenovo", category="Laptops",
        )
        b = P(
            "PC Portable Gamer LENOVO LOQ 15IRX9 i5 13ème Gén 16Go RTX 4050",
            brand="Lenovo", category="Pc Portable",
        )
        assert decide(a, b) == "accept"
        assert overlap(a, b) >= 0.75

    def test_same_laptop_different_storage(self):
        """Same model, different SSD capacity → different variant."""
        a = P(
            "Lenovo LOQ 15IRX9 i5-13450HX 16GB RTX 4050 512GB SSD",
            brand="Lenovo", category="Laptops",
        )
        b = P(
            "Lenovo LOQ 15IRX9 i5-13450HX 16GB RTX 4050 1TB SSD",
            brand="Lenovo", category="Laptops",
        )
        assert decide(a, b) == "reject"
        assert "capacity" in reason(a, b)

    def test_same_laptop_different_ram(self):
        """16GB vs 32GB RAM → same model, different SKU."""
        a = P(
            "Asus TUF Gaming F15 i7-13620H 16GB RTX 4060 512GB",
            brand="Asus", category="Laptops",
        )
        b = P(
            "Asus TUF Gaming F15 i7-13620H 32GB RTX 4060 512GB",
            brand="Asus", category="Laptops",
        )
        # 16GB and 32GB should be caught by capacity mismatch
        # if the extractor picks it up; if it treats "16gb" as a
        # modifier, they reject; if it's just a token, they accept
        # (different products). Currently the extractor puts RAM in
        # capacity — good.
        assert decide(a, b) == "reject"

    def test_different_gpu_tier(self):
        """RTX 4050 vs RTX 4060 → different products."""
        a = P(
            "Lenovo LOQ 15IRX9 i5-13450HX 16GB RTX 4050 512GB",
            brand="Lenovo", category="Laptops",
        )
        b = P(
            "Lenovo LOQ 15IRX9 i5-13450HX 16GB RTX 4060 512GB",
            brand="Lenovo", category="Laptops",
        )
        assert decide(a, b) == "reject"

    def test_different_screen_size_same_model(self):
        """15.6" vs 17.3" → different variants."""
        a = P(
            'Asus ROG Strix G16 16" i7-13650HX RTX 4060 16GB 1TB',
            brand="Asus", category="Laptops",
        )
        b = P(
            'Asus ROG Strix G18 18" i7-13650HX RTX 4060 16GB 1TB',
            brand="Asus", category="Laptops",
        )
        # G16 vs G18 → different models, decisive tokens differ
        assert decide(a, b) == "reject"


# ═══════════════════════════════════════════════════════
# CATEGORY 2 — SMARTPHONES with color and storage
# ═══════════════════════════════════════════════════════
class TestSmartphones:
    def test_same_iphone_different_color(self):
        """Noir vs Blanc → different variants."""
        a = P(
            "iPhone 15 Pro Max 256GB Noir",
            brand="Apple", category="Smartphones",
        )
        b = P(
            "iPhone 15 Pro Max 256Go Blanc",
            brand="Apple", category="Smartphones",
        )
        assert decide(a, b) == "accept"
        assert "color" in reason(a, b)


    def test_same_iphone_different_color(self):
        """
        Noir vs Blanc → SAME product, different cosmetic finish.

        Color is not an identity-defining attribute. Two listings of the
        same iPhone model with different colors are the same SKU at
        different finishes and should be grouped for price comparison.
        """
        a = P(
            "iPhone 15 Pro Max 256GB Noir",
            brand="Apple", category="Smartphones",
        )
        b = P(
            "iPhone 15 Pro Max 256Go Blanc",
            brand="Apple", category="Smartphones",
        )
        assert decide(a, b) == "accept"
        assert "color" not in reason(a, b).lower()

    def test_same_iphone_different_storage(self):
        """256GB vs 512GB → different variants."""
        a = P(
            "iPhone 15 Pro Max 256GB Noir",
            brand="Apple", category="Smartphones",
        )
        b = P(
            "iPhone 15 Pro Max 512GB Noir",
            brand="Apple", category="Smartphones",
        )
        assert decide(a, b) == "reject"
        assert "capacity" in reason(a, b)

    def test_same_iphone_across_sites(self):
        """Same exact variant, different site naming."""
        a = P(
            "Apple iPhone 15 Pro Max 256 Go Titane Naturel",
            brand="Apple", category="Smartphones",
        )
        b = P(
            "iPhone 15 Pro Max 256GB Natural Titanium",
            brand="Apple", category="Smartphones",
        )
        assert decide(a, b) == "accept"

    def test_pro_vs_pro_max(self):
        """iPhone 15 Pro vs Pro Max → different models."""
        a = P(
            "iPhone 15 Pro 256GB Noir",
            brand="Apple", category="Smartphones",
        )
        b = P(
            "iPhone 15 Pro Max 256GB Noir",
            brand="Apple", category="Smartphones",
        )
        # Both have "pro" and "max" and "iphone 15"
        # But Pro vs Pro Max are genuinely different
        # This is the hard case — ideally reject
        assert decide(a, b) == "reject"

    def test_samsung_galaxy_generation(self):
        """Galaxy S23 vs S24 → different generations."""
        a = P(
            "Samsung Galaxy S24 Ultra 12Go 512Go Titanium Black",
            brand="Samsung", category="Smartphones",
        )
        b = P(
            "Samsung Galaxy S23 Ultra 12Go 512Go Noir",
            brand="Samsung", category="Smartphones",
        )
        # s24 vs s23 decisive tokens differ
        assert decide(a, b) == "reject"

    def test_samsung_same_across_sites(self):
        """Same Samsung model, different naming."""
        a = P(
            "Smartphone SAMSUNG Galaxy A55 5G 8Go 256Go Awesome Navy",
            brand="Samsung", category="Smartphones",
        )
        b = P(
            "Samsung Galaxy A55 5G 8GB 256GB - Bleu Marine",
            brand="Samsung", category="Smartphones",
        )
        assert decide(a, b) == "accept"

    def test_xiaomi_redmi_note_different_number(self):
        """Redmi Note 12 vs Note 13 → different generations."""
        a = P(
            "Xiaomi Redmi Note 12 Pro 8Go 256Go Noir",
            brand="Xiaomi", category="Smartphones",
        )
        b = P(
            "Xiaomi Redmi Note 13 Pro 8Go 256Go Noir",
            brand="Xiaomi", category="Smartphones",
        )
        assert decide(a, b) == "reject"

class TestColorIsNotIdentity:
    """
    Colour is cosmetic. Two listings of the same product in different
    colours are the same product at the same price bracket.
    """

    def test_iphone_color_does_not_reject(self):
        a = P("iPhone 15 Pro Max 256GB Noir", brand="Apple", category="Smartphones")
        b = P("iPhone 15 Pro Max 256GB Blanc", brand="Apple", category="Smartphones")
        assert decide(a, b) == "accept"

    def test_laptop_color_does_not_reject(self):
        a = P(
            "Lenovo LOQ 15IRX9 i5-13450HX 16GB RTX 4050 512GB Gris",
            brand="Lenovo", category="Laptops",
        )
        b = P(
            "Lenovo LOQ 15IRX9 i5-13450HX 16GB RTX 4050 512GB Noir",
            brand="Lenovo", category="Laptops",
        )
        assert decide(a, b) == "accept"

    def test_case_color_does_not_reject(self):
        a = P("Coque iPhone 15 Pro Silicone MagSafe Bleu", brand="Apple", category="Coques")
        b = P("Coque iPhone 15 Pro Silicone MagSafe Noir", brand="Apple", category="Coques")
        assert decide(a, b) == "accept"

    def test_color_and_capacity_still_rejects(self):
        """Colour alone doesn't reject, but capacity mismatch does."""
        a = P("iPhone 15 Pro Max 256GB Noir", brand="Apple", category="Smartphones")
        b = P("iPhone 15 Pro Max 512GB Blanc", brand="Apple", category="Smartphones")
        assert decide(a, b) == "reject"
        assert "capacity" in reason(a, b).lower()




class TestPriceGate:
    """
    Price ratio is a hard gate. Two products priced >2× apart are
    not the same product, regardless of name similarity.
    """

    def test_ps4_silicone_cover_vs_controller(self):
        """15 DT silicone cover vs 55 DT controller — different products."""
        a = SimpleNamespace(
            name="SILICON manette PS4",
            brand=None, category="Manettes", price=15.0,
            short_description=None, site="SiteA",
        )
        b = SimpleNamespace(
            name="Manette Sans Fil DualShock 4 adaptable pour PS4",
            brand=None, category="Manettes", price=55.0,
            short_description=None, site="SiteB",
        )
        d = compare_products(a=a, b=b)
        assert d.decision == "reject"
        assert "price" in d.reason.lower()

    def test_ps4_controller_vs_original(self):
        """55 DT generic vs 119 DT original — different products."""
        a = SimpleNamespace(
            name="Manette Sans Fil DualShock 4 adaptable pour PS4",
            brand=None, category="Manettes", price=55.0,
            short_description=None, site="SiteA",
        )
        b = SimpleNamespace(
            name="Manette Sony PS4 DualShock 4 Wireless Controller - Cristale",
            brand="Sony", category="Manettes", price=119.0,
            short_description=None, site="SiteB",
        )
        d = compare_products(a=a, b=b)
        assert d.decision == "reject"
        assert "price" in d.reason.lower()

    def test_ps4_silicone_vs_original(self):
        """15 DT cover vs 119 DT original — extreme price ratio."""
        a = SimpleNamespace(
            name="SILICON manette PS4",
            brand=None, category="Manettes", price=15.0,
            short_description=None, site="SiteA",
        )
        b = SimpleNamespace(
            name="Manette Sony PS4 DualShock 4 Wireless Controller - Cristale",
            brand="Sony", category="Manettes", price=119.0,
            short_description=None, site="SiteB",
        )
        d = compare_products(a=a, b=b)
        assert d.decision == "reject"

    def test_same_product_small_price_diff(self):
        """Same product, 20% price difference — normal cross-site variation."""
        a = SimpleNamespace(
            name="iPhone 15 Pro Max 256GB Noir",
            brand="Apple", category="Smartphones", price=5199.0,
            short_description=None, site="SiteA",
        )
        b = SimpleNamespace(
            name="iPhone 15 Pro Max 256GB Noir",
            brand="Apple", category="Smartphones", price=5249.0,
            short_description=None, site="SiteB",
        )
        d = compare_products(a=a, b=b)
        assert d.decision == "accept"

    def test_same_product_borderline_price_diff(self):
        """Same product, 80% price difference — still acceptable."""
        a = SimpleNamespace(
            name="Lenovo LOQ 15IRX9 i5-13450HX RTX 4050 512GB",
            brand="Lenovo", category="Laptops", price=2500.0,
            short_description=None, site="SiteA",
        )
        b = SimpleNamespace(
            name="Lenovo LOQ 15IRX9 i5-13450HX RTX 4050 512GB",
            brand="Lenovo", category="Laptops", price=3200.0,
            short_description=None, site="SiteB",
        )
        d = compare_products(a=a, b=b)
        assert d.decision == "accept"   # ratio 1.28, well under 2.0

    def test_missing_price_does_not_gate(self):
        """If one price is missing, don't reject on price alone."""
        a = SimpleNamespace(
            name="iPhone 15 Pro Max 256GB Noir",
            brand="Apple", category="Smartphones", price=None,
            short_description=None, site="SiteA",
        )
        b = SimpleNamespace(
            name="iPhone 15 Pro Max 256GB Noir",
            brand="Apple", category="Smartphones", price=5249.0,
            short_description=None, site="SiteB",
        )
        d = compare_products(a=a, b=b)
        assert d.decision == "accept"
# ═══════════════════════════════════════════════════════
# CATEGORY 3 — ACCESSORIES (cases, screen protectors)
# ═══════════════════════════════════════════════════════
class TestAccessories:
    def test_screen_protector_different_phone(self):
        """Same screen protector brand, different target phone."""
        a = P(
            "Film de Protection NEO Pour SAMSUNG A01",
            brand="Neo", category="Film de protection pour Smartphones",
        )
        b = P(
            "Film de protection Nano Glass 9H pour Tecno Pop 2 S Pro",
            brand="Neo", category="Film de protection pour Smartphones",
        )
        assert decide(a, b) == "reject"

    def test_case_same_phone_across_sites(self):
        """Same iPhone case, different site."""
        a = P(
            "Coque iPhone 13 Transparente Silicone Souple",
            brand="Generic", category="Coques",
        )
        b = P(
            "Housse Protection iPhone 13 Crystal Clear Silicone",
            brand="Generic", category="Coques",
        )
        assert decide(a, b) == "accept"

    def test_case_different_iphone_generation(self):
        """Case for iPhone 13 vs iPhone 14 → different fits."""
        a = P(
            "Coque iPhone 13 Transparente",
            brand="Generic", category="Coques",
        )
        b = P(
            "Coque iPhone 14 Transparente",
            brand="Generic", category="Coques",
        )
        # Tokens: coque iphone 13 transparent vs coque iphone 14 transparent
        # Only 13 vs 14 differ — both alphanumeric → decisive token reject
        assert decide(a, b) == "reject"

    def test_charger_different_wattage(self):
        """65W vs 100W charger → different products."""
        a = P(
            "Chargeur GaN 65W USB-C 3 Ports",
            brand="Generic", category="Chargeurs",
        )
        b = P(
            "Chargeur GaN 100W USB-C 3 Ports",
            brand="Generic", category="Chargeurs",
        )
        assert decide(a, b) == "reject"

    def test_charger_same_across_sites(self):
        """Same charger, different naming."""
        a = P(
            "Chargeur Rapide 65W GaN 3 Ports USB-C USB-A",
            brand="Generic", category="Chargeurs",
        )
        b = P(
            "Chargeur GaN 65W USB-C + USB-A x2",
            brand="Generic", category="Chargeurs",
        )
        assert decide(a, b) == "accept"


# ═══════════════════════════════════════════════════════
# CATEGORY 4 — CABLES (length, spec)
# ═══════════════════════════════════════════════════════
class TestCables:
    def test_hdmi_same_across_sites(self):
        """HDMI cable, same specs."""
        a = P(
            "Câble HDMI 2.1 8K 3 mètres Noir",
            brand="Generic", category="Câbles",
        )
        b = P(
            "Cable HDMI 2.1 3M - Noir 8K 60Hz",
            brand="Generic", category="Câbles",
        )
        assert decide(a, b) == "accept"

    def test_hdmi_different_version(self):
        """HDMI 2.0 vs 2.1 → different products."""
        a = P(
            "Câble HDMI 2.1 3 mètres Noir",
            brand="Generic", category="Câbles",
        )
        b = P(
            "Câble HDMI 2.0 3 mètres Noir",
            brand="Generic", category="Câbles",
        )
        # decisive tokens: hdmi 2.1 vs hdmi 2.0 differ
        assert decide(a, b) == "reject"

    def test_hdmi_different_length(self):
        """1m vs 3m → different SKU (usually)."""
        a = P(
            "Câble HDMI 2.1 1 mètre",
            brand="Generic", category="Câbles",
        )
        b = P(
            "Câble HDMI 2.1 3 mètres",
            brand="Generic", category="Câbles",
        )
        assert decide(a, b) == "reject"

    def test_usb_c_same_across_sites(self):
        """USB-C cable, same specs."""
        a = P(
            "Câble USB-C vers USB-C 100W 2 mètres Blanc",
            brand="Generic", category="Câbles",
        )
        b = P(
            "Cable Type-C vers Type-C 100W 2M - Blanc",
            brand="Generic", category="Câbles",
        )
        assert decide(a, b) == "accept"


# ═══════════════════════════════════════════════════════
# CATEGORY 5 — HOME APPLIANCES (capacity, colors)
# ═══════════════════════════════════════════════════════
class TestAppliances:
    def test_fridge_same_across_sites(self):
        """Same fridge, different naming."""
        a = P(
            "Réfrigérateur Samsung RT47CG6002S9EL 463 Litres NoFrost Inox",
            brand="Samsung", category="Réfrigérateurs",
        )
        b = P(
            "Refrigerateur Samsung RT47CG6002S9EL 463L No Frost Inox",
            brand="Samsung", category="Refrigerateurs",
        )
        assert decide(a, b) == "accept"

    def test_fridge_different_capacity(self):
        """400L vs 500L → different products."""
        a = P(
            "Réfrigérateur Samsung NoFrost 400 Litres Inox",
            brand="Samsung", category="Réfrigérateurs",
        )
        b = P(
            "Réfrigérateur Samsung NoFrost 500 Litres Inox",
            brand="Samsung", category="Réfrigérateurs",
        )
        # Numbers 400 vs 500 both appear as tokens → reject
        assert decide(a, b) == "reject"

    def test_washing_machine_same(self):
        """Same washing machine, different naming."""
        a = P(
            "Machine à Laver Frontale Brandt Inverter 9 Kg 1400Tr Silver",
            brand="Brandt", category="Machines à Laver",
        )
        b = P(
            "Lave-linge BRANDT 9KG 1400 tours Inverter Silver",
            brand="Brandt", category="Lave-linge",
        )
        assert decide(a, b) == "accept"

    def test_microwave_different_wattage(self):
        """800W vs 1000W → different products."""
        a = P(
            "Four Micro-Ondes Samsung 800W 23L",
            brand="Samsung", category="Micro-ondes",
        )
        b = P(
            "Four Micro-Ondes Samsung 1000W 23L",
            brand="Samsung", category="Micro-ondes",
        )
        assert decide(a, b) == "reject"


# ═══════════════════════════════════════════════════════
# CATEGORY 6 — FURNITURE
# ═══════════════════════════════════════════════════════
class TestFurniture:
    def test_dining_table_same(self):
        """Same furniture set, different naming."""
        a = P(
            "Pack Salle à Manger SOTUFAB Table SERENA + 6 Chaises SPOT",
            brand="Sotufab", category="Meuble",
        )
        b = P(
            "Ensemble Salle a Manger SOTUFAB SERENA + 6 Chaises SPOT",
            brand="Sotufab", category="Meuble",
        )
        assert decide(a, b) == "accept"

    def test_dining_table_different_chairs(self):
        """Same table, different chair count."""
        a = P(
            "Pack Salle à Manger SOTUFAB Table SERENA + 6 Chaises",
            brand="Sotufab", category="Meuble",
        )
        b = P(
            "Pack Salle à Manger SOTUFAB Table SERENA + 4 Chaises",
            brand="Sotufab", category="Meuble",
        )
        # 6 vs 4 differ
        assert decide(a, b) == "reject"


# ═══════════════════════════════════════════════════════
# CATEGORY 7 — COOLING / PC COMPONENTS
# ═══════════════════════════════════════════════════════
class TestPCComponents:
    def test_deepcool_different_models(self):
        """LQ360 vs LT360 → different products (DeepCool case)."""
        a = P(
            "DeepCool LQ360 Ultra AIO 360mm | LCD | Black",
            brand="DeepCool", category="Watercooling",
        )
        b = P(
            "DeepCool LT360 VISION ARGB 360mm | LCD | Black",
            brand="DeepCool", category="Watercooling",
        )
        assert decide(a, b) == "reject"
        assert "decisive" in reason(a, b).lower() or "token" in reason(a, b).lower()

    def test_deepcool_same_model(self):
        """Same DeepCool model, different site."""
        a = P(
            "Watercooling DeepCool LQ360 Ultra 360mm Noir",
            brand="DeepCool", category="Watercooling",
        )
        b = P(
            "DeepCool LQ360 ULTRA AIO 360mm | LCD | Black",
            brand="DeepCool", category="Watercooling",
        )
        assert decide(a, b) == "accept"

    def test_gpu_same_model(self):
        """Same GPU, different site."""
        a = P(
            "Carte Graphique PNY GeForce RTX 5070 12Go OC Triple Fan",
            brand="PNY", category="Composants Informatique",
        )
        b = P(
            "PNY RTX 5070 12GB OC Triple Fan GDDR7",
            brand="PNY", category="Cartes Graphiques",
        )
        assert decide(a, b) == "accept"

    def test_gpu_different_memory(self):
        """12GB vs 16GB → different products."""
        a = P(
            "Carte Graphique PNY GeForce RTX 5070 12Go OC",
            brand="PNY", category="GPU",
        )
        b = P(
            "Carte Graphique PNY GeForce RTX 5070 16Go OC",
            brand="PNY", category="GPU",
        )
        assert decide(a, b) == "reject"

    def test_motherboard_same(self):
        """Same motherboard, different naming."""
        a = P(
            "Carte mère MSI PRO B860-P WIFI / ATX / LGA 1851 / DDR5",
            brand="MSI", category="Cartes mères",
        )
        b = P(
            "Carte Mere MSI B860-P WIFI ATX LGA1851 DDR5",
            brand="MSI", category="Cartes-meres",
        )
        assert decide(a, b) == "accept"

    def test_ram_different_speed(self):
        """3200MHz vs 3600MHz → different products."""
        a = P(
            "Barrette Mémoire Dato 16 Go DDR4 3200MHz U-Dimm",
            brand="Dato", category="Barrette-mémoire",
        )
        b = P(
            "Barrette Mémoire Dato 16 Go DDR4 3600MHz U-Dimm",
            brand="Dato", category="Barrette-mémoire",
        )
        # 3200 vs 3600 decisive tokens
        assert decide(a, b) == "reject"


# ═══════════════════════════════════════════════════════
# EDGE CASES — same product, radically different naming
# ═══════════════════════════════════════════════════════
class TestEdgeCases:
    def test_audi_radio_same_model(self):
        """Same radio, different site naming."""
        a = P(
            "Radio & Réveil Avec Projection Thomson CP202T Bois",
            brand="Thomson", category="Radio Réveil",
        )
        b = P(
            "Thomson CP202T Radio Reveil Projection Bois",
            brand="Thomson", category="Radio Reveil",
        )
        assert decide(a, b) == "accept"

    def test_earbuds_same_model(self):
        """Same earbuds, different naming."""
        a = P(
            "Écouteurs Sans Fil JBL Soundgear Clips / Noir",
            brand="JBL", category="Airpods JBL",
        )
        b = P(
            "JBL SoundGear Clips True Wireless Noir",
            brand="JBL", category="Ecouteurs",
        )
        assert decide(a, b) == "accept"

    def test_tablet_same(self):
        """Same tablet, different naming."""
        a = P(
            'TABLETTE OSCAL PAD 200 13.4" / 6Go/128Go / 4G/WIFI / Gris +keyboard +Pen',
            brand="Oscal", category="Tablettes Tactiles",
        )
        b = P(
            "OSCAL PAD 200 6/128GB Gris + Clavier + Stylus",
            brand="Oscal", category="Tablettes",
        )
        assert decide(a, b) == "accept"

    def test_empty_brands_both_sides(self):
        """Two listings with no brand should not auto-reject on brand."""
        a = P(
            "Câble HDMI 2.1 8K 3 mètres Noir",
            brand=None, category="Câbles",
        )
        b = P(
            "Cable HDMI 2.1 3M Noir 8K",
            brand=None, category="Câbles",
        )
        assert decide(a, b) == "accept"

    def test_one_side_missing_brand(self):
        """One side has no brand — don't gate on brand."""
        a = P(
            "Coque iPhone 13 Transparente",
            brand="Polo", category="Coques",
        )
        b = P(
            "Housse iPhone 13 Crystal Clear",
            brand=None, category="Coques",
        )
        assert decide(a, b) == "accept"

    def test_no_url_no_crash(self):
        """Both products missing URL — rules should still work."""
        a = P("Test Product 123", brand="Test", category="Test")
        b = P("Test Product 123", brand="Test", category="Test")
        assert decide(a, b) == "accept"