# Prompts for material property extraction

MATERIAL_PROMPT = """
Extract material filling additives from the uploaded documents. 
Please identify and list all filler materials used in the polymers or composites described.
For each filler, include the name and any mentioned properties or concentration if available.
"""

MATERIAL_NAME_PROMPT = """
Extract primary polymer material information from the uploaded documents.
Identify the base polymer materials used in the products or applications described.
For each polymer, provide the full name, common abbreviation if available, and any key properties mentioned.
"""

GENDER_PROMPT = """
Determine connector gender information from the uploaded documents.
For each connector mentioned, identify whether it is male, female, or hermaphroditic/hybrid.
Include the connector name, part number if available, and the gender classification.
""" 