"""TCGA _primary_disease -> CCLE-aligned cancer_type mapping (for CSV annotation only)."""

# Former "na" TCGA diseases mapped to CCLE primary_disease-style names.
# These should be listed in config/pretrain_cancer_type_exclude.json (not used in training).
CCLE_STYLE_FILLED_FROM_NA = {
    "acute myeloid leukemia": "Leukemia",
    "adrenocortical cancer": "Adrenal Cancer",
    "adrenocortical carcinoma": "Adrenal Cancer",
    "diffuse large b-cell lymphoma": "Lymphoma",
    "pheochromocytoma & paraganglioma": "Adrenal Cancer",
    "pheochromocytoma and paraganglioma": "Adrenal Cancer",
    "testicular germ cell tumor": "Teratoma",
    "testicular germ cell tumors": "Teratoma",
    "chronic myelogenous leukemia": "Leukemia",
}

# Default exclude list when filling CSV (also set in pretrain_cancer_type_exclude.json).
DEFAULT_EXCLUDE_CANCER_TYPES = sorted(
    {"na"} | set(CCLE_STYLE_FILLED_FROM_NA.values())
)


def norm_disease_name(v: str) -> str:
    return str(v).strip().lower().replace("&", "and")


def build_tcga_name_to_cancer_type_map() -> dict:
    """Map normalized TCGA disease name to CCLE-aligned cancer_type string."""
    study_to_source_map = {
        "LAML": "Leukemia",
        "ACC": "Adrenal Cancer",
        "BLCA": "Bladder Cancer",
        "LGG": "Brain Cancer",
        "BRCA": "Breast Cancer",
        "CESC": "Cervical Cancer",
        "CHOL": "Bile Duct Cancer",
        "LCML": "Leukemia",
        "COAD": "Colon/Colorectal Cancer",
        "CNTL": "na",
        "ESCA": "Esophageal Cancer",
        "FPPP": "na",
        "GBM": "Brain Cancer",
        "HNSC": "Head and Neck Cancer",
        "KICH": "Kidney Cancer",
        "KIRC": "Kidney Cancer",
        "KIRP": "Kidney Cancer",
        "LIHC": "Liver Cancer",
        "LUAD": "Lung Cancer",
        "LUSC": "Lung Cancer",
        "DLBC": "Lymphoma",
        "MESO": "na",
        "MISC": "na",
        "OV": "Ovarian Cancer",
        "PAAD": "Pancreatic Cancer",
        "PCPG": "Adrenal Cancer",
        "PRAD": "Prostate Cancer",
        "READ": "Colon/Colorectal Cancer",
        "SARC": "Sarcoma",
        "SKCM": "Skin Cancer",
        "STAD": "Gastric Cancer",
        "TGCT": "Teratoma",
        "THYM": "na",
        "THCA": "Thyroid Cancer",
        "UCS": "Endometrial/Uterine Cancer",
        "UCEC": "Endometrial/Uterine Cancer",
        "UVM": "Eye Cancer",
    }
    study_name_to_abbr = {
        "Acute Myeloid Leukemia": "LAML",
        "Adrenocortical carcinoma": "ACC",
        "Bladder Urothelial Carcinoma": "BLCA",
        "Brain Lower Grade Glioma": "LGG",
        "Breast invasive carcinoma": "BRCA",
        "Cervical squamous cell carcinoma and endocervical adenocarcinoma": "CESC",
        "Cholangiocarcinoma": "CHOL",
        "Chronic Myelogenous Leukemia": "LCML",
        "Colon adenocarcinoma": "COAD",
        "Controls": "CNTL",
        "Esophageal carcinoma": "ESCA",
        "FFPE Pilot Phase II": "FPPP",
        "Glioblastoma multiforme": "GBM",
        "Head and Neck squamous cell carcinoma": "HNSC",
        "Kidney Chromophobe": "KICH",
        "Kidney renal clear cell carcinoma": "KIRC",
        "Kidney renal papillary cell carcinoma": "KIRP",
        "Liver hepatocellular carcinoma": "LIHC",
        "Lung adenocarcinoma": "LUAD",
        "Lung squamous cell carcinoma": "LUSC",
        "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma": "DLBC",
        "Mesothelioma": "MESO",
        "Miscellaneous": "MISC",
        "Ovarian serous cystadenocarcinoma": "OV",
        "Pancreatic adenocarcinoma": "PAAD",
        "Pheochromocytoma and Paraganglioma": "PCPG",
        "Prostate adenocarcinoma": "PRAD",
        "Rectum adenocarcinoma": "READ",
        "Sarcoma": "SARC",
        "Skin Cutaneous Melanoma": "SKCM",
        "Stomach adenocarcinoma": "STAD",
        "Testicular Germ Cell Tumors": "TGCT",
        "Thymoma": "THYM",
        "Thyroid carcinoma": "THCA",
        "Uterine Carcinosarcoma": "UCS",
        "Uterine Corpus Endometrial Carcinoma": "UCEC",
        "Uveal Melanoma": "UVM",
    }
    name_to_type = {
        norm_disease_name(name): study_to_source_map[abbr]
        for name, abbr in study_name_to_abbr.items()
        if abbr in study_to_source_map
    }
    target_to_source_map = {
        "acute myeloid leukemia": "Leukemia",
        "adrenocortical cancer": "Adrenal Cancer",
        "bladder urothelial carcinoma": "Bladder Cancer",
        "brain lower grade glioma": "Brain Cancer",
        "breast invasive carcinoma": "Breast Cancer",
        "cervical & endocervical cancer": "Cervical Cancer",
        "cholangiocarcinoma": "Bile Duct Cancer",
        "colon adenocarcinoma": "Colon/Colorectal Cancer",
        "diffuse large b-cell lymphoma": "Lymphoma",
        "esophageal carcinoma": "Esophageal Cancer",
        "glioblastoma multiforme": "Brain Cancer",
        "head & neck squamous cell carcinoma": "Head and Neck Cancer",
        "kidney chromophobe": "Kidney Cancer",
        "kidney clear cell carcinoma": "Kidney Cancer",
        "kidney papillary cell carcinoma": "Kidney Cancer",
        "liver hepatocellular carcinoma": "Liver Cancer",
        "lung adenocarcinoma": "Lung Cancer",
        "lung squamous cell carcinoma": "Lung Cancer",
        "mesothelioma": "na",
        "ovarian serous cystadenocarcinoma": "Ovarian Cancer",
        "pancreatic adenocarcinoma": "Pancreatic Cancer",
        "pheochromocytoma & paraganglioma": "Adrenal Cancer",
        "prostate adenocarcinoma": "Prostate Cancer",
        "rectum adenocarcinoma": "Colon/Colorectal Cancer",
        "sarcoma": "Sarcoma",
        "skin cutaneous melanoma": "Skin Cancer",
        "stomach adenocarcinoma": "Gastric Cancer",
        "testicular germ cell tumor": "Teratoma",
        "thymoma": "na",
        "thyroid carcinoma": "Thyroid Cancer",
        "uterine carcinosarcoma": "Endometrial/Uterine Cancer",
        "uterine corpus endometrioid carcinoma": "Endometrial/Uterine Cancer",
        "uveal melanoma": "Eye Cancer",
    }
    name_to_type.update({norm_disease_name(k): v for k, v in target_to_source_map.items()})
    name_to_type.update({
        "pheochromocytoma and paraganglioma": "Adrenal Cancer",
        "head and neck squamous cell carcinoma": "Head and Neck Cancer",
        "cervical and endocervical cancer": "Cervical Cancer",
        "adrenocortical carcinoma": "Adrenal Cancer",
        "diffuse large b-cell lymphoma": "Lymphoma",
        "kidney renal clear cell carcinoma": "Kidney Cancer",
        "kidney renal papillary cell carcinoma": "Kidney Cancer",
        "testicular germ cell tumors": "Teratoma",
        "uterine corpus endometrial carcinoma": "Endometrial/Uterine Cancer",
    })
    return name_to_type


def map_primary_disease_to_cancer_type(primary_disease: str, name_to_type: dict) -> str:
    """Return mapped cancer_type; unmapped diseases become 'na'."""
    mapped = name_to_type.get(norm_disease_name(primary_disease))
    return mapped if mapped is not None else "na"
