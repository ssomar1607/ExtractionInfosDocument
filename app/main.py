text = """DOCUMENT D’INFORMATIONS CLÉS
Objectif : Le présent document contient des informations essentielles sur le produit d’investissement. Il ne s’agit pas d’un
document à caractère commercial.
Ces informations vous sont fournies conformément à une obligation légale, aﬁn de vous aider à comprendre en quoi consiste ce
produit et quels risques, coûts, gains et pertes potentiels y sont associés, et de vous aider à le comparer à d’autres produits.
Produit
AMUNDI FUNDS EUROPEAN EQUITY INCOME ESG - A2 EUR MTI
Un Compartiment d’Amundi Funds
LU1883311570 - Devise : EUR
Ce Compartiment est agréé au Luxembourg. 
Société de gestion : Amundi Luxembourg S.A. (ci-après : « nous »), membre du groupe Amundi, est agréé au Luxembourg et réglementé par la
Commission de surveillance du secteur ﬁnancier (CSSF)."""


if str("LU1883311570 - Devise : EUR")  in str(text):
    print("yes")
else:
    print("no")