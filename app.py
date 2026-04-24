"""
GOAT — Gene Ontology & Analysis Tool
Streamlit front-end for the gene research agent pipeline.
"""

import streamlit as st
import time, re, html as html_lib, xml.etree.ElementTree as ET, os, collections, json
from datetime import datetime
from io import StringIO, BytesIO
import requests
import pandas as pd
from openai import OpenAI

# ── Session state initialisation ──────────────────────────────────────────────
# All pipeline outputs live here so they survive Streamlit re-runs
# (which happen on every widget interaction, including download button clicks).
for _key in ("results", "go_bp_data", "reactome_data",
             "synthesis_results", "synthesis_tables", "pipeline_done", "run_metrics"):
    if _key not in st.session_state:
        st.session_state[_key] = [] if _key in ("results","go_bp_data","reactome_data") \
                                  else ({} if _key in ("synthesis_results","synthesis_tables","run_metrics") \
                                  else False)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GOAT — Gene Research Agent",
    page_icon="🐐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🐐 GOAT — Gene Ontology & Analysis Tool")
st.caption("Comprehensive literature mining + optional downstream enrichment")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — LLM provider + configuration
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── LLM provider ─────────────────────────────────────────────────────────
    st.subheader("LLM Provider")

    PROVIDER_REGISTRY = {
        "groq":       {"base_url": "https://api.groq.com/openai/v1",          "model": "llama-3.3-70b-versatile",                "label": "Groq — Llama 3.3 70B (Free)"},
        "openrouter": {"base_url": "https://openrouter.ai/api/v1",            "model": "meta-llama/llama-3.3-70b-instruct:free", "label": "OpenRouter — Llama 3.3 70B (Free)"},
        "anthropic":  {"base_url": "https://api.anthropic.com/v1",            "model": "claude-sonnet-4-5",                      "label": "Anthropic — Claude Sonnet (Paid)"},
        "openai":     {"base_url": "https://api.openai.com/v1",               "model": "gpt-4o-mini",                            "label": "OpenAI — GPT-4o mini (Paid)"},
        "AKI":        {"base_url": "https://aki.io/v1",                       "model": "gpt-oss-120b",                           "label": "AKI — gpt-oss-120b (Paid)"},
        "denbi-gpt":  {"base_url": "https://denbi-llm-api.bihealth.org/v1",   "model": "gpt-oss-120b",                           "label": "deNBI — gpt-oss-120b"},
        "denbi-qwen": {"base_url": "https://denbi-llm-api.bihealth.org/v1",   "model": "qwen3.5-fp8",                            "label": "deNBI — qwen3.5-fp8"},
        "denbi-m25":  {"base_url": "https://denbi-llm-api.bihealth.org/v1",   "model": "m25-reap-172b",                          "label": "deNBI — m25-reap-172b"},
        "ollama":     {"base_url": "http://localhost:11434/v1",                "model": "llama3",                                 "label": "Ollama — local (Free)"},
        "other":      {"base_url": None,                                       "model": None,                                     "label": "Other (custom)"},
    }

    provider_labels = {k: v["label"] for k, v in PROVIDER_REGISTRY.items()}
    selected_provider_key = st.selectbox(
        "Provider",
        options=list(PROVIDER_REGISTRY.keys()),
        format_func=lambda k: provider_labels[k],
    )

    if selected_provider_key == "other":
        custom_base_url = st.text_input("Base URL", placeholder="https://your-provider.com/v1")
        custom_model    = st.text_input("Model name", placeholder="e.g. mistral-7b-instruct")
        LLM_BASE_URL    = custom_base_url
        LLM_MODEL       = custom_model
    elif selected_provider_key == "ollama":
        LLM_BASE_URL = PROVIDER_REGISTRY["ollama"]["base_url"]
        LLM_MODEL    = PROVIDER_REGISTRY["ollama"]["model"]
    else:
        LLM_BASE_URL = PROVIDER_REGISTRY[selected_provider_key]["base_url"]
        LLM_MODEL    = PROVIDER_REGISTRY[selected_provider_key]["model"]

    if selected_provider_key == "ollama":
        API_KEY = "ollama"
        st.info("Ollama selected — no API key required.")
    else:
        API_KEY = st.text_input("API Key", type="password", placeholder="Paste your API key here")

    PROVIDER = selected_provider_key

    st.divider()

    # ── Core inputs ───────────────────────────────────────────────────────────
    st.subheader("Core Inputs")

    genes_raw = st.text_area(
        "Gene list *",
        placeholder="COL1A1, COL1A2, CTHRC1 ...\n(comma- or newline-separated)",
        height=100,
    )

    organism_options = {
        "Homo sapiens":    "Human (Homo sapiens)",
        "Mus musculus":    "Mouse (Mus musculus)",
        "__other__":       "Other...",
    }
    selected_organism = st.selectbox(
        "Organism *",
        options=list(organism_options.keys()),
        format_func=lambda k: organism_options[k],
    )
    if selected_organism == "__other__":
        UNIPROT_ORGANISM = st.text_input("Scientific name", placeholder="e.g. Rattus norvegicus")
    else:
        UNIPROT_ORGANISM = selected_organism

    ANALYSIS_NAME = st.text_input("Analysis name *", placeholder="e.g. liver_fibrosis_screen")

    st.divider()

    # ── Optional context ──────────────────────────────────────────────────────
    st.subheader("Optional Context")
    DISEASE_CONTEXT = st.text_input("Disease context", placeholder="e.g. fibrosis")
    ORGAN_CONTEXT   = st.text_input("Organ / tissue context", placeholder="e.g. liver")
    OTHER_CONTEXT   = st.text_input("Other context", placeholder="e.g. splicing, transcriptomics")
    NOTES           = st.text_area("Notes", placeholder="Documentation notes — sample metadata, experimental rationale, caveats, etc.", height=100, key="notes_field")

    st.divider()

    # ── PubMed search parameters ──────────────────────────────────────────────
    st.subheader("PubMed Parameters")
    MAX_ABSTRACTS_PER_GENE = st.slider("Max abstracts per gene", 1, 50, 5)
    YEAR_FROM = st.number_input("Year from", min_value=1900, max_value=2100, value=2020, step=1)
    fallback_raw = st.text_input("Fallback year steps", value="2015, 2010, 2005, 2000")
    try:
        FALLBACK_STEPS = [int(x.strip()) for x in fallback_raw.split(",") if x.strip()]
    except ValueError:
        FALLBACK_STEPS = [2015, 2010, 2005, 2000]
        st.warning("Fallback steps should be comma-separated years.")

    st.divider()

    # ── Database module toggles ──────────────────────────────────────────────
    st.subheader("Database Modules")
    st.caption("Select the databases you wish to query.")

    ENABLE_GO_PATHWAYS         = st.checkbox("GO Pathways + Reactome", value=True)
    ENABLE_DRUG_TARGETS        = st.checkbox("Drug Targets",           value=True)
    ENABLE_DISEASE_ASSOCIATION = st.checkbox("Disease Associations",   value=True)
    ENABLE_CLINICAL_TRIALS     = st.checkbox("Clinical Trials",        value=True)
    ENABLE_GWAS                = st.checkbox("GWAS Catalog",           value=True)

    st.divider()

    # ── LLM synthesis module toggles ──────────────────────────────────────────
    st.subheader("LLM Synthesis Modules")
    st.caption("Select the synthesis modules you wish to run. Note that the Literature Synthesis module will always run.")
    RUN_9A = st.checkbox("Literature Synthesis",  value=True, key="run_9a")
    _therapeutic_modules_enabled = ENABLE_DRUG_TARGETS or ENABLE_DISEASE_ASSOCIATION or ENABLE_CLINICAL_TRIALS or ENABLE_GWAS
    RUN_9B = st.checkbox("Therapeutic Landscape", value=_therapeutic_modules_enabled, disabled=not (RUN_9A and _therapeutic_modules_enabled), key="run_9b")
    RUN_9C = st.checkbox("Experimental Proposal", value=RUN_9A and RUN_9B,            disabled=not (RUN_9A and RUN_9B), key="run_9c")
    RUN_9D = st.checkbox("Target Prioritisation", value=RUN_9A and RUN_9B and RUN_9C, disabled=not (RUN_9A and RUN_9B and RUN_9C), key="run_9d")
    if not _therapeutic_modules_enabled:
        st.caption("⚠️ Therapeutic Landscape requires at least one of: Drug Targets, Disease Associations, Clinical Trials, or GWAS Catalog.")

    st.divider()

    # ── Scope flags ───────────────────────────────────────────────────────────
    st.subheader("Search Scope")
    st.caption("Select which context fields to include in each module's query.")

    has_dis = bool(DISEASE_CONTEXT.strip())
    has_org = bool(ORGAN_CONTEXT.strip())
    has_oth = bool(OTHER_CONTEXT.strip())

    # Column header row
    _scope_header = st.columns([2, 1, 1, 1, 1])
    _scope_header[0].markdown("<span style='font-size:0.78em; color:gray; text-transform:uppercase; letter-spacing:0.05em;'>Module</span>", unsafe_allow_html=True)
    for _i, _lbl in enumerate(["Disease", "Organ", "Organism", "Other"], start=1):
        _scope_header[_i].markdown(f"<span style='font-size:0.78em; color:gray; text-transform:uppercase; letter-spacing:0.05em;'>{_lbl}</span>", unsafe_allow_html=True)

    # PubMed row
    _pub_cols = st.columns([2, 1, 1, 1, 1])
    _pub_cols[0].markdown("**PubMed**")
    pubmed_d = _pub_cols[1].checkbox(" ", value=has_dis, disabled=not has_dis, key="pub_d", label_visibility="collapsed")
    pubmed_o = _pub_cols[2].checkbox(" ", value=False,   disabled=not has_org, key="pub_o", label_visibility="collapsed")
    pubmed_r = _pub_cols[3].checkbox(" ", value=False,                          key="pub_r", label_visibility="collapsed")
    pubmed_x = _pub_cols[4].checkbox(" ", value=False,   disabled=not has_oth, key="pub_x", label_visibility="collapsed")
    if pubmed_r:
        st.warning("Organism filter may significantly reduce PubMed results due to sparse MeSH annotation.", icon="⚠️")

    PUBMED_SCOPE = {"disease": pubmed_d, "organ": pubmed_o, "organism": pubmed_r, "other": pubmed_x}

    # ClinicalTrials row
    _ct_cols = st.columns([2, 1, 1, 1, 1])
    _ct_cols[0].markdown("**Clinical Trials**")
    ct_d = _ct_cols[1].checkbox(" ", value=has_dis, disabled=not has_dis, key="ct_d", label_visibility="collapsed")
    ct_o = _ct_cols[2].checkbox(" ", value=False,   disabled=not has_org, key="ct_o", label_visibility="collapsed")
    _ct_cols[3].markdown("<span style='color:#bbb; padding-left:8px;'>—</span>", unsafe_allow_html=True)
    ct_x = _ct_cols[4].checkbox(" ", value=False,   disabled=not has_oth, key="ct_x", label_visibility="collapsed")
    CT_SCOPE = {"disease": ct_d, "organ": ct_o, "organism": False, "other": ct_x}

    # LLM Synthesis row
    _llm_cols = st.columns([2, 1, 1, 1, 1])
    _llm_cols[0].markdown("**LLM Synthesis**")
    llm_d = _llm_cols[1].checkbox(" ", value=has_dis, disabled=not has_dis, key="llm_d", label_visibility="collapsed")
    llm_o = _llm_cols[2].checkbox(" ", value=False,   disabled=not has_org, key="llm_o", label_visibility="collapsed")
    _llm_cols[3].markdown("<span style='color:#bbb; padding-left:8px;'>—</span>", unsafe_allow_html=True)
    llm_x = _llm_cols[4].checkbox(" ", value=False,   disabled=not has_oth, key="llm_x", label_visibility="collapsed")
    LLM_SCOPE = {"disease": llm_d, "organ": llm_o, "organism": False, "other": llm_x}

    st.divider()

    # ── Session save / load ───────────────────────────────────────────────────
    st.subheader("💾 Session")
    st.caption("Save results and reload them later, even after closing the browser.")

    if st.session_state.pipeline_done:
        _session_payload = {
            "results":           st.session_state.results,
            "go_bp_data":        st.session_state.go_bp_data,
            "reactome_data":     st.session_state.reactome_data,
            "synthesis_results": st.session_state.synthesis_results,
            "synthesis_tables":  st.session_state.synthesis_tables,
            "saved_at":          datetime.now().isoformat(),
        }
        st.download_button(
            label="⬇️ Save session (.json)",
            data=json.dumps(_session_payload, indent=2, default=str),
            file_name=f"goat_session_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            key="session_save_btn",
        )

    _uploaded_session = st.file_uploader("Resume session (.json)", type="json", key="session_loader")
    if _uploaded_session is not None:
        try:
            _loaded = json.loads(_uploaded_session.read())
            st.session_state.results           = _loaded.get("results", [])
            st.session_state.go_bp_data        = _loaded.get("go_bp_data", [])
            st.session_state.reactome_data     = _loaded.get("reactome_data", [])
            st.session_state.synthesis_results = _loaded.get("synthesis_results", {})
            st.session_state.synthesis_tables  = _loaded.get("synthesis_tables", {})
            st.session_state.pipeline_done     = True
            st.success(f"Session restored — {len(st.session_state.results)} gene(s) loaded.")
        except Exception as e:
            st.error(f"Could not load session: {e}")

    st.divider()
    run_button = st.button("▶ Run GOAT", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
NCBI_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_EMAIL = "research-agent@notebook.local"
TODAY_YEAR = datetime.now().year
MIN_HITS_BEFORE_FALLBACK = 3

GO_EXPERIMENTAL_CODES = {
    "EXP","IDA","IPI","IMP","IGI","IEP",
    "HTP","HDA","HMP","HGI","HEP",
    "TAS","IC","NAS"
}

STRIP_PATTERNS = [
    r"\bproprotein\b", r"\bprecursor\b", r"\bprepro\w*\b", r"\bpreprotein\b",
    r"\bputative\b", r"\bisoform\s+\w+\b", r"\bsplice\s+variant\b",
    r"\bshort\s+form\b", r"\blong\s+form\b", r"\bhomolog(?:ue)?\b",
    r"\d+\s*kDa\b", r"\((?:human|mouse|rat|bovine|murine|porcine|canine)\)",
    r",?\s*subunit\s+\w+$", r",?\s*chain\s+\w+$",
]

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS  (identical logic to notebook, display via st.write)
# ══════════════════════════════════════════════════════════════════════════════

def clean_protein_name(name):
    if not name:
        return name
    cleaned = name
    for pat in STRIP_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,-")


def fetch_ncbi_gene(gene, species, log, max_retries=3, base_sleep=1.0):
    try:
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(
                    f"{NCBI_BASE}/esearch.fcgi",
                    params={"db": "gene", "term": f"{gene}[Gene Name] AND {species}[Organism]",
                            "retmode": "json", "retmax": 1, "email": NCBI_EMAIL},
                    timeout=10
                )
                r.raise_for_status()
                break
            except Exception as e:
                if attempt == max_retries: raise
                time.sleep(base_sleep * (2 ** (attempt - 1)))

        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            log(f"    NCBI Gene: no results for {gene}")
            return None, None, None, None
        gene_id = ids[0]
        time.sleep(0.35)

        for attempt in range(1, max_retries + 1):
            try:
                r2 = requests.get(
                    f"{NCBI_BASE}/efetch.fcgi",
                    params={"db": "gene", "id": gene_id, "retmode": "xml", "email": NCBI_EMAIL},
                    timeout=20
                )
                r2.raise_for_status()
                break
            except Exception as e:
                if attempt == max_retries: raise
                time.sleep(base_sleep * (2 ** (attempt - 1)))

        root      = ET.fromstring(r2.content)
        name_el   = root.find(".//Gene-ref_desc")
        full_name = name_el.text.strip() if name_el is not None and name_el.text else None
        summ_el   = root.find(".//Entrezgene_summary")
        summary   = summ_el.text.strip() if summ_el is not None and summ_el.text else None

        ensembl_gene_id = None
        for dbtag in root.findall(".//Dbtag"):
            db_el = dbtag.find("Dbtag_db")
            if db_el is not None and db_el.text == "Ensembl":
                oid_el = dbtag.find(".//Object-id_str")
                if oid_el is not None and oid_el.text:
                    val = oid_el.text.strip()
                    if val.startswith("ENSG"):
                        ensembl_gene_id = val.split(".")[0]
                        break

        return gene_id, full_name, summary, ensembl_gene_id
    except Exception as e:
        log(f"    NCBI Gene lookup failed for {gene}: {e}")
        return None, None, None, None


def resolve_uniprot(gene, organism, ensembl_from_ncbi=None, log=print):
    try:
        url    = "https://rest.uniprot.org/uniprotkb/search"
        params = {"query": f"gene:{gene} AND organism_name:{organism} AND reviewed:true",
                  "format": "json", "size": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        hits = r.json().get("results", [])
        if not hits:
            params["query"] = f"gene:{gene} AND organism_name:{organism}"
            r    = requests.get(url, params=params, timeout=10)
            hits = r.json().get("results", [])
        if not hits:
            return None, None, ensembl_from_ncbi, [], []

        entry     = hits[0]
        accession = entry["primaryAccession"]
        pn        = entry.get("proteinDescription", {})
        rec       = pn.get("recommendedName", {})
        raw_name  = rec.get("fullName", {}).get("value")
        if not raw_name:
            sub      = pn.get("submissionNames", [])
            raw_name = sub[0].get("fullName", {}).get("value") if sub else None
        prot_name = clean_protein_name(raw_name)

        ensembl_from_uniprot = None
        go_bp_terms          = []
        reactome_paths       = []

        for xref in entry.get("uniProtKBCrossReferences", []):
            db = xref.get("database", "")
            if db == "Ensembl" and not ensembl_from_uniprot:
                for prop in xref.get("properties", []):
                    if prop.get("key") == "GeneId":
                        ensembl_from_uniprot = prop["value"].split(".")[0]
                        break
            elif db == "GO":
                props    = {p["key"]: p["value"] for p in xref.get("properties", [])}
                go_term  = props.get("GoTerm", "")
                evidence = props.get("GoEvidenceType", "")
                aspect   = go_term.split(":")[0].strip() if ":" in go_term else ""
                term     = go_term.split(":", 1)[-1].strip() if ":" in go_term else go_term
                ev_code  = evidence.split(":")[0].strip() if ":" in evidence else evidence
                if aspect == "P" and ev_code in GO_EXPERIMENTAL_CODES:
                    go_bp_terms.append({"go_id": xref.get("id", ""), "go_term": term, "evidence": ev_code})
            elif db == "Reactome":
                r_id   = xref.get("id", "")
                r_name = next((p["value"] for p in xref.get("properties", [])
                               if p.get("key") == "PathwayName"), None)
                if r_id and r_name:
                    reactome_paths.append({"reactome_id": r_id, "pathway_name": r_name})

        ensembl_gene_id = ensembl_from_ncbi or ensembl_from_uniprot
        return accession, prot_name, ensembl_gene_id, go_bp_terms, reactome_paths
    except Exception as e:
        log(f"    UniProt lookup failed for {gene}: {e}")
        return None, None, ensembl_from_ncbi, [], []


def search_pubmed_with_fallback(gene, official_full_name, max_results, year_from,
                                min_hits, fallback_steps,
                                PUBMED_SCOPE, DISEASE_CONTEXT, ORGAN_CONTEXT,
                                UNIPROT_ORGANISM, OTHER_CONTEXT, log=print):
    gene_term = f"{gene}[Title/Abstract]"
    if (official_full_name
            and official_full_name.lower() != gene.lower()
            and gene.lower() not in official_full_name.lower()):
        gene_term = f'({gene}[Title/Abstract] OR "{official_full_name}"[Title/Abstract])'

    context_terms = []
    if PUBMED_SCOPE.get("disease") and DISEASE_CONTEXT:
        context_terms.append(f"{DISEASE_CONTEXT}[Title/Abstract]")
    if PUBMED_SCOPE.get("organ") and ORGAN_CONTEXT:
        context_terms.append(f"{ORGAN_CONTEXT}[Title/Abstract]")
    if PUBMED_SCOPE.get("organism"):
        context_terms.append(f'"{UNIPROT_ORGANISM}"[Organism]')
    if PUBMED_SCOPE.get("other") and OTHER_CONTEXT:
        context_terms.append(f"{OTHER_CONTEXT}[Title/Abstract]")

    context_str = (" AND " + " AND ".join(context_terms)) if context_terms else ""
    windows     = [year_from] + fallback_steps

    for idx, start_year in enumerate(windows):
        query  = f"{gene_term}{context_str} AND {start_year}:{TODAY_YEAR}[pdat]"
        params = {"db": "pubmed", "term": query, "retmax": max_results,
                  "retmode": "json", "sort": "relevance", "email": NCBI_EMAIL}
        try:
            r             = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=15)
            r.raise_for_status()
            ids           = r.json()["esearchresult"]["idlist"]
            window_label  = f"{start_year}-{TODAY_YEAR}"
            fallback_note = "" if idx == 0 else f" (fallback from {year_from})"
            if len(ids) >= min_hits or idx == len(windows) - 1:
                return ids, window_label, fallback_note
            log(f"    Only {len(ids)} hit(s) for {start_year}-{TODAY_YEAR}, widening...")
            time.sleep(0.35)
        except Exception as e:
            log(f"    PubMed search error: {e}")
            return [], f"{start_year}-{TODAY_YEAR}", ""
    return [], f"{fallback_steps[-1]}-{TODAY_YEAR}", ""


def extract_year(art):
    for xp in [".//ArticleDate/Year",
               ".//PubMedPubDate[@PubStatus='pubmed']/Year",
               ".//PubDate/Year",
               ".//PubMedPubDate[@PubStatus='medline']/Year",
               ".//PubDate/MedlineDate"]:
        el = art.find(xp)
        if el is not None and el.text:
            return el.text.strip()[:4]
    return "n.d."


def fetch_abstracts(pmids, log=print):
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids),
              "retmode": "xml", "rettype": "abstract", "email": NCBI_EMAIL}
    try:
        r    = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        arts = []
        for art in root.findall(".//PubmedArticle"):
            try:
                pmid_el  = art.find(".//PMID")
                pmid     = pmid_el.text.strip() if pmid_el is not None else ""
                title_el = art.find(".//ArticleTitle")
                title    = "".join(title_el.itertext()).strip() if title_el is not None else ""
                ab_parts = []
                for ab in art.findall(".//AbstractText"):
                    lbl  = ab.get("Label", "")
                    text = "".join(ab.itertext()).strip()
                    if text:
                        ab_parts.append(f"{lbl + ': ' if lbl else ''}{text}")
                abstract = " ".join(ab_parts)
                year     = extract_year(art)
                jt_el    = art.find(".//Journal/Title") or art.find(".//ISOAbbreviation")
                journal  = jt_el.text.strip() if jt_el is not None else "Unknown"
                if pmid and (title or abstract):
                    arts.append({"pmid": pmid, "title": title, "abstract": abstract,
                                 "year": year, "journal": journal})
            except Exception:
                continue
        return arts
    except Exception as e:
        log(f"    Abstract fetch failed: {e}")
        return []


def fetch_icite(pmids, log=print):
    if not pmids:
        return {}
    try:
        r = requests.get("https://icite.od.nih.gov/api/pubs",
                         params={"pmids": ",".join(pmids)}, timeout=15)
        r.raise_for_status()
        return {str(p["pmid"]): p.get("citation_count", 0) for p in r.json().get("data", [])}
    except Exception as e:
        log(f"    iCite fetch failed: {e}")
        return {}


def truncate_to_sentences(text, max_chars=1000):
    if len(text) <= max_chars:
        return text
    truncated      = text[:max_chars]
    last_boundary  = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    return truncated[:last_boundary + 1] if last_boundary > 0 else truncated


def build_context_desc(LLM_SCOPE, DISEASE_CONTEXT, ORGAN_CONTEXT, OTHER_CONTEXT):
    context_parts = []
    if LLM_SCOPE.get("disease") and DISEASE_CONTEXT:
        core = DISEASE_CONTEXT
        if LLM_SCOPE.get("organ") and ORGAN_CONTEXT:
            core += f" ({ORGAN_CONTEXT})"
        context_parts.append(core)
    if LLM_SCOPE.get("other") and OTHER_CONTEXT:
        context_parts.append(f"other: {OTHER_CONTEXT}")
    return " | ".join(context_parts) if context_parts else "a biological context"


def fmt_runtime(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def build_core_system(context_desc):
    return (
        f"You are a biomedical research analyst specialising in {context_desc}, extracting gene-level mechanistic findings from primary literature to orient a research team.\n\n"
        "CONSTRAINTS:\n"
        "- The gene is always the grammatical subject — each sentence must describe what the gene does, not what the study did.\n"
        "- Mechanistic findings take priority over clinical or epidemiological ones.\n"
        "- Only use PMIDs from the provided abstracts. Never invent citations.\n"
        "- Use the RefSeq summary only as background — do not restate it.\n\n"
        "TASK: Write a single synthesis paragraph integrating the provided PubMed abstracts into a mechanistic narrative about the gene.\n\n"
        "OUTPUT FORMAT:\n"
        "- Write one sentence per distinct finding. If two abstracts support the same finding, merge them into one sentence.\n"
        "- End every sentence with its PMID citation(s): [PMID: XXXXX] or [PMID: XXXXX, PMID: YYYYY].\n"
        "- Do not begin the paragraph with the gene name or a summary sentence.\n"
        "- Output the paragraph only — no title, no preamble, no JSON, no markdown."
    )


def synthesise_literature(gene, protein_name, refseq_summary, articles,
                           llm, LLM_MODEL, MAX_ABSTRACTS_PER_GENE,
                           LLM_SCOPE, DISEASE_CONTEXT, ORGAN_CONTEXT, OTHER_CONTEXT,
                           log=print):
    if not articles:
        return "No abstracts available for synthesis.", 0

    context_desc = build_context_desc(LLM_SCOPE, DISEASE_CONTEXT, ORGAN_CONTEXT, OTHER_CONTEXT)
    parts = []
    for a in articles[:MAX_ABSTRACTS_PER_GENE]:
        parts.append(
            f"PMID {a['pmid']} ({a['year']}, {a['journal']}, {a['citations']} citations)\n"
            f"Title: {a['title']}\n"
            f"Abstract: {truncate_to_sentences(a['abstract'])}"
        )
    context  = "\n\n---\n\n".join(parts)
    name_str = f"{gene} ({protein_name})" if protein_name else gene
    refseq_ctx = f"RefSeq summary: {refseq_summary}\n\n" if refseq_summary else ""
    user_msg   = (
        f"Gene/protein: {name_str}\n"
        + (f"Context: {context_desc}\n" if context_desc != "a biological context" else "")
        + f"\n{refseq_ctx}"
        + f"Abstracts (ranked highest citations first):\n{context}"
    )
    try:
        resp   = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": build_core_system(context_desc)},
                      {"role": "user",   "content": user_msg}],
            temperature=0.1, timeout=60)
        text   = resp.choices[0].message.content.strip()
        tokens = resp.usage.total_tokens if resp.usage else 0
        return text, tokens
    except Exception as e:
        log(f"    LLM synthesis failed for {gene}: {e}")
        return "LLM synthesis failed.", 0


def substitute_pmid_refs(text, pmid_to_idx):
    def replace(m):
        pmid = m.group(1)
        return f"[{pmid_to_idx[pmid]}]" if pmid in pmid_to_idx else m.group(0)
    return re.sub(r"\[PMID:\s*(\d+)\]", replace, text)


# ── OpenTargets ───────────────────────────────────────────────────────────────
OT_URL   = "https://api.platform.opentargets.org/api/v4/graphql"
OT_QUERY = """
query targetInfo($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    associatedDiseases(page: { index: 0, size: 5 }) {
      rows {
        score
        disease { name }
      }
    }
  }
}
"""

def query_opentargets(ensembl_id, log=print):
    try:
        r = requests.post(OT_URL, json={"query": OT_QUERY, "variables": {"ensemblId": ensembl_id}}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            target = (data.get("data") or {}).get("target")
            return target
        return (data.get("data") or {}).get("target")
    except Exception as e:
        log(f"    OT query failed: {e}")
        return None


def parse_diseases(target_data):
    if not target_data:
        return "N/A"
    dis_rows = (target_data.get("associatedDiseases") or {}).get("rows") or []
    if not dis_rows:
        return "None found"
    return "; ".join(
        f"{d['disease']['name']} ({round(d['score'], 3)})"
        for d in dis_rows[:5]
        if d.get("disease") and d.get("score") is not None
    ) or "None found"


# ── ChEMBL helpers ────────────────────────────────────────────────────────────
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

def get_json_chembl(resource, params=None, timeout=45, log=print):
    try:
        r = requests.get(f"{CHEMBL_BASE}/{resource}.json", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"    ChEMBL request failed: {e}")
        return None

def extract_records(payload, preferred_keys):
    if not isinstance(payload, dict):
        return []
    for k in preferred_keys:
        v = payload.get(k)
        if isinstance(v, list):
            return v
    for v in payload.values():
        if isinstance(v, list):
            return v
    return []

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def fetch_chembl_targets(uniprot_acc, log=print):
    data    = get_json_chembl("target", params={"target_components__accession": uniprot_acc, "limit": 20}, log=log)
    records = extract_records(data, ["targets"])
    out     = []
    for rec in records:
        comps     = rec.get("target_components") or []
        comp_accs = []
        for c in comps:
            acc = (c.get("accession") or c.get("component_accession") or "").strip()
            if acc:
                comp_accs.append(acc.split("-")[0])
        out.append({
            "target_chembl_id": rec.get("target_chembl_id") or "",
            "pref_name":        rec.get("pref_name") or "",
            "target_type":      rec.get("target_type") or "",
            "organism":         rec.get("organism") or "",
            "accessions":       sorted(set(comp_accs)),
        })
    return [x for x in out if x["target_chembl_id"]]

def fetch_chembl_mechanisms(target_chembl_id, log=print):
    data    = get_json_chembl("mechanism", params={"target_chembl_id": target_chembl_id, "limit": 200}, log=log)
    records = extract_records(data, ["mechanisms"])
    out     = []
    for rec in records:
        molecule_id = rec.get("molecule_chembl_id") or rec.get("drug_chembl_id") or ""
        if not molecule_id:
            continue
        out.append({
            "molecule_chembl_id":  molecule_id,
            "target_chembl_id":    rec.get("target_chembl_id") or target_chembl_id,
            "mechanism_of_action": rec.get("mechanism_of_action") or "",
            "action_type":         rec.get("action_type") or "",
            "max_phase":           rec.get("max_phase"),
        })
    return out

def fetch_molecule_details(molecule_ids, log=print):
    details = {}
    for chunk in chunked(sorted(set(molecule_ids)), 50):
        data    = get_json_chembl("molecule", params={"molecule_chembl_id__in": ",".join(chunk), "limit": max(50, len(chunk))}, log=log)
        records = extract_records(data, ["molecules"])
        for rec in records:
            mid = rec.get("molecule_chembl_id") or ""
            if mid:
                details[mid] = {"pref_name": rec.get("pref_name") or "", "molecule_type": rec.get("molecule_type") or "", "max_phase": rec.get("max_phase")}
    return details

def fetch_indications(molecule_ids, DISEASE_CONTEXT, log=print):
    from collections import defaultdict
    disease_context = (DISEASE_CONTEXT or "").strip().lower()
    exact_hits = defaultdict(set)
    all_hits   = defaultdict(set)
    for chunk in chunked(sorted(set(molecule_ids)), 40):
        data    = get_json_chembl("drug_indication", params={"molecule_chembl_id__in": ",".join(chunk), "limit": 1000}, log=log)
        records = extract_records(data, ["drug_indications"])
        for rec in records:
            mid   = rec.get("molecule_chembl_id") or rec.get("parent_molecule_chembl_id") or ""
            label = rec.get("mesh_heading") or rec.get("efo_term") or rec.get("indication") or ""
            if not mid or not label:
                continue
            all_hits[mid].add(label)
            if disease_context and disease_context in label.lower():
                exact_hits[mid].add(label)
    out = {}
    for mid in sorted(set(molecule_ids)):
        labels   = sorted(exact_hits[mid]) if exact_hits[mid] else sorted(all_hits[mid])
        out[mid] = "; ".join(labels[:5]) if labels else "N/A"
    return out

def phase_to_status(phase):
    if phase in (None, "", "N/A"): return "N/A"
    try:
        p = int(float(phase))
    except Exception:
        return "N/A"
    if p >= 4: return "Approved"
    if p >= 1: return f"Clinical Phase {p}"
    return "Preclinical/unknown"

def _phase_num(phase):
    try: return int(float(phase))
    except Exception: return 0

def base_molecule_name(name):
    name = name.strip().upper()
    name = re.sub(
        r"\s+(HYDROCHLORIDE|MALEATE|TARTRATE|FUMARATE|CITRATE|PHOSPHATE|"
        r"LACTATE|SULFATE|SULPHATE|ACETATE|BROMIDE|CHLORIDE|MESYLATE|"
        r"SODIUM|POTASSIUM|CALCIUM|MAGNESIUM|ZINC)$",
        "", name)
    return name.strip()

def build_drug_rows(gene, mechanisms, molecule_details, indication_map):
    grouped = {}
    for rec in mechanisms:
        mid = rec.get("molecule_chembl_id") or ""
        if not mid: continue
        mol      = molecule_details.get(mid, {})
        phase    = mol.get("max_phase") or rec.get("max_phase")
        status   = phase_to_status(phase)
        raw_name = (mol.get("pref_name") or mid).upper()
        base     = base_molecule_name(raw_name)
        if base not in grouped:
            grouped[base] = {"gene": gene, "molecule": mol.get("pref_name") or mid,
                             "drug_type": mol.get("molecule_type") or "N/A",
                             "status": status, "indication": indication_map.get(mid, "N/A"),
                             "_phase_num": _phase_num(phase)}
        elif _phase_num(phase) > grouped[base]["_phase_num"]:
            grouped[base].update({"molecule": mol.get("pref_name") or mid,
                                  "drug_type": mol.get("molecule_type") or "N/A",
                                  "status": status, "indication": indication_map.get(mid, "N/A"),
                                  "_phase_num": _phase_num(phase)})
    all_drugs = list(grouped.values())
    approved  = [d for d in all_drugs if d["status"] == "Approved"]
    clinical  = sorted([d for d in all_drugs if d["status"].startswith("Clinical")],
                       key=lambda x: x["_phase_num"], reverse=True)
    selected  = approved[:2] + clinical[:3]
    for d in selected: d.pop("_phase_num", None)
    return selected


# ── ClinicalTrials ────────────────────────────────────────────────────────────
CT_BASE = "https://clinicaltrials.gov/api/v2/studies"

def build_ct_query(gene, protein_name, CT_SCOPE, DISEASE_CONTEXT, ORGAN_CONTEXT,
                   UNIPROT_ORGANISM, OTHER_CONTEXT):
    if (protein_name and protein_name != "N/A"
            and gene.lower() not in protein_name.lower()
            and protein_name.lower() not in gene.lower()):
        base = f'({gene} OR "{protein_name}")'
    else:
        base = gene
    context_parts = []
    if CT_SCOPE.get("disease") and DISEASE_CONTEXT: context_parts.append(DISEASE_CONTEXT)
    if CT_SCOPE.get("organ")   and ORGAN_CONTEXT:   context_parts.append(ORGAN_CONTEXT)
    if CT_SCOPE.get("other")   and OTHER_CONTEXT:   context_parts.append(OTHER_CONTEXT)
    return (base + " " + " ".join(context_parts)).strip()

def search_clinical_trials(gene, protein_name, CT_SCOPE, DISEASE_CONTEXT,
                           ORGAN_CONTEXT, UNIPROT_ORGANISM, OTHER_CONTEXT,
                           max_results=5, log=print):
    query = build_ct_query(gene, protein_name, CT_SCOPE, DISEASE_CONTEXT,
                           ORGAN_CONTEXT, UNIPROT_ORGANISM, OTHER_CONTEXT)
    try:
        r = requests.get(CT_BASE, params={
            "query.term": query,
            "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING",
            "pageSize": max_results, "format": "json"}, timeout=20)
        r.raise_for_status()
        trials = []
        for s in r.json().get("studies", []):
            proto    = s.get("protocolSection", {})
            ident    = proto.get("identificationModule", {})
            stat     = proto.get("statusModule", {})
            design   = proto.get("designModule", {})
            sponsor  = proto.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
            outcomes = proto.get("outcomesModule", {}).get("primaryOutcomes", [])
            trials.append({
                "nct_id":           ident.get("nctId", ""),
                "title":            ident.get("briefTitle", "")[:80],
                "phase":            "; ".join(design.get("phases", [])) or "N/A",
                "status":           stat.get("overallStatus", ""),
                "sponsor":          sponsor.get("name", "")[:40],
                "primary_endpoint": outcomes[0].get("measure", "")[:60] if outcomes else "N/A",
                "est_completion":   stat.get("primaryCompletionDateStruct", {}).get("date", "N/A"),
            })
        return trials
    except Exception as e:
        log(f"    ClinicalTrials query failed for {gene}: {e}")
        return []


# ── GWAS ──────────────────────────────────────────────────────────────────────
GWAS_BASE = "https://www.ebi.ac.uk/gwas/rest/api"

def fetch_href(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def search_gwas(gene, MAX_SNPS=5, MAX_ASSOCS_PER_SNP=3, log=print):
    try:
        r = requests.get(
            f"{GWAS_BASE}/singleNucleotidePolymorphisms/search/findByGene",
            params={"geneName": gene, "size": MAX_SNPS}, timeout=15)
        r.raise_for_status()
        snps = r.json().get("_embedded", {}).get("singleNucleotidePolymorphisms", [])
    except Exception as e:
        log(f"    GWAS SNP lookup failed for {gene}: {e}")
        return []

    seen_rsids, hits = set(), []
    for snp in snps:
        rsid = snp.get("rsId", "")
        if rsid in seen_rsids: continue
        seen_rsids.add(rsid)
        assocs     = []
        assoc_href = snp.get("_links", {}).get("associations", {}).get("href", "")
        if assoc_href:
            data   = fetch_href(assoc_href)
            assocs = (data or {}).get("_embedded", {}).get("associations", [])
        if not assocs:
            hits.append({"rsid": rsid, "trait": "N/A", "p_value": "N/A",
                         "effect_size": "N/A", "beta_direction": "N/A", "study_accession": "N/A"})
            continue
        for assoc in assocs[:MAX_ASSOCS_PER_SNP]:
            assoc_links = assoc.get("_links", {})
            efo_href    = assoc_links.get("efoTraits", {}).get("href", "")
            traits      = []
            if efo_href:
                efo_data = fetch_href(efo_href)
                efo_list = (efo_data or {}).get("_embedded", {}).get("efoTraits", [])
                traits   = [t.get("trait", "") for t in efo_list]
            trait_str   = "; ".join(filter(None, traits)) or "N/A"
            study_href  = assoc_links.get("study", {}).get("href", "")
            accession   = "N/A"
            if study_href:
                study_data = fetch_href(study_href)
                accession  = (study_data or {}).get("accessionId", "N/A")
            beta   = assoc.get("betaNum")
            OR     = assoc.get("orPerCopyNum")
            effect = f"{beta} {assoc.get('betaUnit', '')}".strip() if beta else (OR or "N/A")
            hits.append({"rsid": rsid, "trait": trait_str, "p_value": assoc.get("pvalue", "N/A"),
                         "effect_size": effect, "beta_direction": assoc.get("betaDirection", "N/A"),
                         "study_accession": accession})
            time.sleep(0.15)
    return hits[:MAX_SNPS]


# ── Sankey (Plotly) ───────────────────────────────────────────────────────────
def build_sankey_fig(df, label_col, title, min_shared=2,
                     gene_color="#4E79A7", term_color="#F28E2B", max_terms=40):
    import plotly.graph_objects as go_plot
    if df.empty:
        return None
    term_counts  = df.groupby(label_col)["gene"].nunique()
    shared_terms = term_counts[term_counts >= min_shared].nlargest(max_terms).index.tolist()
    if not shared_terms:
        shared_terms = term_counts.nlargest(min(max_terms, len(term_counts))).index.tolist()
    df_s         = df[df[label_col].isin(shared_terms)]
    genes        = sorted(df_s["gene"].unique().tolist())
    terms        = df_s[label_col].unique().tolist()
    nodes        = genes + terms
    node_idx     = {n: i for i, n in enumerate(nodes)}
    node_colors  = [gene_color] * len(genes) + [term_color] * len(terms)
    sources, targets, values, link_labels = [], [], [], []
    for _, row in df_s.iterrows():
        g = row["gene"]; t = row[label_col]; n = term_counts[t]
        sources.append(node_idx[g]); targets.append(node_idx[t]); values.append(1)
        link_labels.append(f"{g} -> {t} (shared by {n} gene{'s' if n > 1 else ''})")
    fig = go_plot.Figure(go_plot.Sankey(
        arrangement="snap",
        node=dict(pad=12, thickness=18, label=nodes, color=node_colors,
                  hovertemplate="%{label}<extra></extra>"),
        link=dict(source=sources, target=targets, value=values,
                  label=link_labels, color="rgba(180,180,180,0.35)")
    ))
    fig.update_layout(title_text=title, title_font_size=14, font_size=11,
                      height=max(450, len(nodes) * 14), margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT — created from sidebar values on every page load, always available
# ══════════════════════════════════════════════════════════════════════════════
# This must live outside `if run_button:` so that the LLM synthesis tab
# can call it after a re-run triggered by a download button click.
llm = OpenAI(api_key=API_KEY if API_KEY else "ollama", base_url=LLM_BASE_URL)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run pipeline when button is clicked
# ══════════════════════════════════════════════════════════════════════════════
if run_button:
    # ── Validation ────────────────────────────────────────────────────────────
    errors = []
    raw_genes = genes_raw.replace("\n", ",")
    GENE_LIST = [g.strip() for g in raw_genes.split(",") if g.strip()]
    if not GENE_LIST:
        errors.append("Gene list is required.")
    if not UNIPROT_ORGANISM or (selected_organism == "__other__" and not UNIPROT_ORGANISM):
        errors.append("Organism is required.")
    if not ANALYSIS_NAME.strip():
        errors.append("Analysis name is required.")
    if selected_provider_key != "ollama" and not API_KEY:
        errors.append("API key is required for this provider.")
    if selected_provider_key == "other" and (not LLM_BASE_URL or not LLM_MODEL):
        errors.append("Custom provider requires Base URL and Model name.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # ── Status area ───────────────────────────────────────────────────────────
    status_box = st.empty()
    log_lines  = []

    def log(msg):
        log_lines.append(msg)
        status_box.code("\n".join(log_lines[-30:]), language=None)

    log(f"GOAT pipeline starting — {len(GENE_LIST)} gene(s)")
    log(f"  Provider : {PROVIDER}  |  Model : {LLM_MODEL}")
    log(f"  Organism : {UNIPROT_ORGANISM}")
    log("-" * 55)

    results        = []
    go_bp_data     = []
    reactome_data  = []
    total_tokens   = 0
    pipeline_start = time.time()

    prog = st.progress(0, text="Starting...")

    for i, gene in enumerate(GENE_LIST):
        prog.progress((i) / len(GENE_LIST), text=f"Processing {gene} ({i+1}/{len(GENE_LIST)})...")
        log(f"\n[{i+1}/{len(GENE_LIST)}] {gene}")

        ncbi_gene_id, official_full_name, refseq_summary, ensembl_from_ncbi = \
            fetch_ncbi_gene(gene, UNIPROT_ORGANISM, log)
        log(f"    NCBI: {ncbi_gene_id or 'not found'} | {official_full_name or 'N/A'} | Ensembl: {ensembl_from_ncbi or 'not found'}")

        uniprot_id, protein_name, ensembl_gene_id, go_bp_terms, reactome_paths = \
            resolve_uniprot(gene, UNIPROT_ORGANISM, ensembl_from_ncbi=ensembl_from_ncbi, log=log)
        log(f"    UniProt: {uniprot_id or 'not found'} | GO BP: {len(go_bp_terms)} | Reactome: {len(reactome_paths)}")

        for term in go_bp_terms:
            go_bp_data.append({"gene": gene, **term})
        for path in reactome_paths:
            reactome_data.append({"gene": gene, **path})

        time.sleep(0.3)

        log(f"    PubMed search...")
        pmids, window, fallback_note = search_pubmed_with_fallback(
            gene, official_full_name, MAX_ABSTRACTS_PER_GENE,
            YEAR_FROM, MIN_HITS_BEFORE_FALLBACK, FALLBACK_STEPS,
            PUBMED_SCOPE, DISEASE_CONTEXT, ORGAN_CONTEXT,
            UNIPROT_ORGANISM, OTHER_CONTEXT, log)
        log(f"    Found {len(pmids)} article(s) [{window}{fallback_note}]")
        time.sleep(0.4)

        base_row = {
            "gene": gene, "ncbi_gene_id": ncbi_gene_id or "N/A",
            "official_full_name": official_full_name or "N/A",
            "refseq_summary": refseq_summary or "N/A",
            "uniprot_id": uniprot_id or "N/A",
            "protein_name": protein_name or "N/A",
            "ensembl_gene_id": ensembl_gene_id or "N/A",
            "go_bp_count": len(go_bp_terms), "reactome_count": len(reactome_paths),
        }

        if not pmids:
            results.append({**base_row, "search_window_used": window,
                            "articles_found": 0, "status": "NO_PUBMED"})
            continue

        log(f"    Fetching abstracts...")
        articles = fetch_abstracts(pmids, log)
        log(f"    Retrieved {len(articles)} abstract(s)")
        time.sleep(0.4)

        if not articles:
            results.append({**base_row, "search_window_used": window,
                            "articles_found": 0, "status": "NO_ABSTRACTS"})
            continue

        log(f"    iCite citation counts...")
        citations = fetch_icite([a["pmid"] for a in articles], log)
        for a in articles:
            a["citations"] = citations.get(a["pmid"], 0)
        articles.sort(key=lambda x: x["citations"], reverse=True)
        log(f"    Ranked by citations (top: {articles[0]['citations']})")
        time.sleep(0.3)

        log(f"    LLM synthesis ({LLM_MODEL})...")
        llm_synthesis, tokens = synthesise_literature(
            gene, protein_name, refseq_summary, articles,
            llm, LLM_MODEL, MAX_ABSTRACTS_PER_GENE,
            LLM_SCOPE, DISEASE_CONTEXT, ORGAN_CONTEXT, OTHER_CONTEXT, log)
        total_tokens += tokens

        pmid_to_idx        = {a["pmid"]: idx + 1 for idx, a in enumerate(articles)}
        llm_summary_papers = substitute_pmid_refs(llm_synthesis, pmid_to_idx)
        references         = "\n".join(
            f"[{idx+1}] {a['year']} · {a['journal']} · {a['pmid']} · {a['citations']} cit"
            for idx, a in enumerate(articles)
        )
        log(f"    Done (tokens: {tokens:,})")

        results.append({
            **base_row,
            "llm_summary_papers": llm_summary_papers,
            "references": references,
            "search_window_used": window + fallback_note,
            "articles_found": len(articles),
            "status": "OK",
        })

    prog.progress(1.0, text="Core pipeline complete ✓")
    core_elapsed = time.time() - pipeline_start
    log(f"\nCore pipeline complete — {sum(1 for r in results if r.get('status')=='OK')} OK | {total_tokens:,} tokens | {fmt_runtime(core_elapsed)}")

    # ── Persist to session_state so downloads don't wipe results ─────────────
    st.session_state.results       = results
    st.session_state.go_bp_data    = go_bp_data
    st.session_state.reactome_data = reactome_data
    st.session_state.pipeline_done = True
    st.session_state.run_metrics   = {
        "core": {"tokens": total_tokens, "runtime": core_elapsed},
    }

    # ══════════════════════════════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ══════════════════════════════════════════════════════════════════════════

# Show results whenever pipeline_done is True (survives re-runs / downloads)
if st.session_state.pipeline_done:
    results       = st.session_state.results
    go_bp_data    = st.session_state.go_bp_data
    reactome_data = st.session_state.reactome_data
    tabs = st.tabs(["📋 Core Results", "🧬 GO & Reactome", "💊 Drugs & Diseases",
                    "🏥 Clinical Trials", "🧪 GWAS", "🤖 LLM Synthesis", "⬇️ Export"])

    # ── Tab 1: Core results ───────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Core Pipeline Results")
        df = pd.DataFrame(results)
        ok = df[df["status"] == "OK"]
        if not ok.empty:
            display_cols = [c for c in ["gene","official_full_name","uniprot_id",
                                         "ensembl_gene_id","articles_found","status"]
                            if c in ok.columns]
            st.dataframe(ok[display_cols], use_container_width=True)
            for _, r in ok.iterrows():
                with st.expander(f"📄 {r['gene']} — {r.get('official_full_name','N/A')}"):
                    st.markdown(f"**RefSeq summary:** {r.get('refseq_summary','N/A')}")
                    st.markdown(f"**Literature synthesis:** {r.get('llm_summary_papers','N/A')}")
                    if r.get("references") and r["references"] not in ("N/A",""):
                        st.markdown("**References:**")
                        for line in r["references"].split("\n"):
                            st.caption(line)

    # ── Tab 2: GO & Reactome ──────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("GO Biological Process & Reactome Pathways")
        if not ENABLE_GO_PATHWAYS:
            st.info("GO Pathways module was disabled.")
        elif not go_bp_data and not reactome_data:
            st.warning("No GO/Reactome data found.")
        else:
            # Pathway summaries
            def build_pathway_summary(gene):
                go_gene_terms    = [d["go_term"]      for d in go_bp_data    if d["gene"] == gene]
                react_gene_paths = [d["pathway_name"] for d in reactome_data if d["gene"] == gene]
                go_counts        = collections.Counter(d["go_term"]      for d in go_bp_data)
                react_counts     = collections.Counter(d["pathway_name"] for d in reactome_data)
                top_go    = [t for t, _ in go_counts.most_common()    if t in go_gene_terms][:5]
                top_react = [p for p, _ in react_counts.most_common() if p in react_gene_paths][:5]
                parts = []
                if top_go:    parts.append("GO BP: " + "; ".join(top_go))
                if top_react: parts.append("Reactome: " + "; ".join(top_react))
                return " | ".join(parts) if parts else "N/A"

            for r in results:
                if r.get("status") == "OK":
                    r["pathway_summary"] = build_pathway_summary(r["gene"])

            df_go       = pd.DataFrame(go_bp_data)    if go_bp_data    else pd.DataFrame()
            df_reactome = pd.DataFrame(reactome_data) if reactome_data else pd.DataFrame()

            context_suffix = DISEASE_CONTEXT or ""
            if ORGAN_CONTEXT:
                context_suffix += f" ({ORGAN_CONTEXT})" if context_suffix else ORGAN_CONTEXT

            if not df_go.empty:
                st.markdown("### GO Biological Process")
                fig = build_sankey_fig(df_go, "go_term",
                    f"GO Biological Process{' — '+context_suffix if context_suffix else ''}",
                    gene_color="#4E79A7", term_color="#F28E2B")
                if fig: st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_go, use_container_width=True)

            if not df_reactome.empty:
                st.markdown("### Reactome Pathways")
                fig = build_sankey_fig(df_reactome, "pathway_name",
                    f"Reactome Pathways{' — '+context_suffix if context_suffix else ''}",
                    gene_color="#4E79A7", term_color="#59A14F")
                if fig: st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_reactome, use_container_width=True)

    # ── Tab 3: Drugs & Diseases ───────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Drug Targets & Disease Associations")
        if not ENABLE_DRUG_TARGETS and not ENABLE_DISEASE_ASSOCIATION:
            st.info("Drug/disease modules were disabled.")
        else:
            drug_rows = []
            ot_rows   = []
            ok_results = [r for r in results if r.get("status") == "OK"]

            cell6_prog = st.progress(0, text="Querying OpenTargets & ChEMBL...")
            for idx_r, r in enumerate(ok_results):
                gene = r["gene"]
                eid  = r.get("ensembl_gene_id", "N/A")
                r.setdefault("chembl_molecules", "None")
                r.setdefault("ot_disease_association", "N/A")

                cell6_prog.progress((idx_r) / max(len(ok_results), 1),
                                    text=f"Querying {gene}...")

                if not eid or eid == "N/A":
                    ot_rows.append({"gene": gene, "top_diseases": "No Ensembl ID"})
                else:
                    if ENABLE_DISEASE_ASSOCIATION:
                        target_data = query_opentargets(eid, log)
                        top_diseases = parse_diseases(target_data)
                        r["ot_disease_association"] = top_diseases
                        ot_rows.append({"gene": gene, "top_diseases": top_diseases})

                    if ENABLE_DRUG_TARGETS:
                        uniprot_acc = r.get("uniprot_id", "N/A")
                        if uniprot_acc and uniprot_acc != "N/A":
                            chembl_targets = fetch_chembl_targets(uniprot_acc, log)
                            all_mechanisms = []
                            seen_keys      = set()
                            for tgt in chembl_targets[:5]:
                                mechs = fetch_chembl_mechanisms(tgt["target_chembl_id"], log)
                                for m in mechs:
                                    k = (m["molecule_chembl_id"], m["mechanism_of_action"])
                                    if k not in seen_keys:
                                        seen_keys.add(k)
                                        all_mechanisms.append(m)
                            if all_mechanisms:
                                mol_ids          = sorted(set(m["molecule_chembl_id"] for m in all_mechanisms if m.get("molecule_chembl_id")))
                                mol_details      = fetch_molecule_details(mol_ids, log)
                                indication_map   = fetch_indications(mol_ids, DISEASE_CONTEXT, log)
                                drugs            = build_drug_rows(gene, all_mechanisms, mol_details, indication_map)
                                if drugs:
                                    drug_rows.extend(drugs)
                                    r["chembl_molecules"] = "; ".join(f"{d['molecule']} [{d['status']}]" for d in drugs)
                time.sleep(0.3)

            cell6_prog.progress(1.0, text="Done ✓")

            if ot_rows:
                st.markdown("#### Disease associations (OpenTargets)")
                st.dataframe(pd.DataFrame(ot_rows), use_container_width=True)

            if drug_rows:
                st.markdown("#### Known drugs (ChEMBL)")
                st.dataframe(pd.DataFrame(drug_rows), use_container_width=True)
            elif ENABLE_DRUG_TARGETS:
                st.info("No drug data found via ChEMBL.")

    # ── Tab 4: Clinical Trials ────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Clinical Trials (ClinicalTrials.gov)")
        if not ENABLE_CLINICAL_TRIALS:
            st.info("Clinical trials module was disabled.")
        else:
            ct_rows  = []
            ct_prog  = st.progress(0, text="Querying ClinicalTrials.gov...")
            ok_ct    = [r for r in results if r.get("status") == "OK"]
            for idx_r, r in enumerate(ok_ct):
                gene         = r["gene"]
                protein_name = clean_protein_name(r.get("protein_name", ""))
                ct_prog.progress((idx_r) / max(len(ok_ct), 1), text=f"Querying {gene}...")
                trials = search_clinical_trials(gene, protein_name, CT_SCOPE,
                                                DISEASE_CONTEXT, ORGAN_CONTEXT,
                                                UNIPROT_ORGANISM, OTHER_CONTEXT, log=log)
                for t in trials:
                    ct_rows.append({"gene": gene, **t})
                r["active_trials"] = len(trials)
                r["trial_phases"]  = "; ".join(set(t["phase"] for t in trials)) if trials else "None"
                time.sleep(0.5)
            ct_prog.progress(1.0, text="Done ✓")
            if ct_rows:
                df_ct = pd.DataFrame(ct_rows)
                st.dataframe(df_ct, use_container_width=True)
            else:
                st.info("No active trials found.")

    # ── Tab 5: GWAS ───────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("GWAS Catalog")
        if not ENABLE_GWAS:
            st.info("GWAS module was disabled.")
        else:
            gwas_rows  = []
            gwas_prog  = st.progress(0, text="Querying GWAS Catalog...")
            ok_gwas    = [r for r in results if r.get("status") == "OK"]
            for idx_r, r in enumerate(ok_gwas):
                gene = r["gene"]
                gwas_prog.progress((idx_r) / max(len(ok_gwas), 1), text=f"Querying {gene}...")
                hits = search_gwas(gene, log=log)
                for h in hits:
                    gwas_rows.append({"gene": gene, **h})
                r["gwas_hits"]   = len(hits)
                r["gwas_traits"] = "; ".join(set(h["trait"] for h in hits if h["trait"] != "N/A")) or "None"
                time.sleep(0.5)
            gwas_prog.progress(1.0, text="Done ✓")
            if gwas_rows:
                df_gwas = pd.DataFrame(gwas_rows)
                st.dataframe(df_gwas, use_container_width=True)
            else:
                st.info("No GWAS associations found.")

    # ── Tab 6: LLM Synthesis ─────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("LLM Synthesis (Sections 9a–9d)")
        if not RUN_9A:
            st.info("LLM synthesis is disabled — enable at least '9A — Literature Synthesis' in the sidebar.")
        else:
            successful = [r for r in results if r.get("status") == "OK"]
            if not successful:
                st.warning("No successful gene results to synthesise.")
            else:
                context_desc = build_context_desc(LLM_SCOPE, DISEASE_CONTEXT, ORGAN_CONTEXT, OTHER_CONTEXT)

                def build_persona_context():
                    spec_parts = []
                    if DISEASE_CONTEXT:
                        spec_parts.append(f"{ORGAN_CONTEXT+' ' if ORGAN_CONTEXT else ''}{DISEASE_CONTEXT}")
                    elif ORGAN_CONTEXT:
                        spec_parts.append(f"{ORGAN_CONTEXT} biology")
                    expert = f"with expert knowledge in {OTHER_CONTEXT}" if OTHER_CONTEXT else ""
                    if spec_parts and expert:
                        return f"specialising in {', '.join(spec_parts)}, {expert}"
                    elif spec_parts:
                        return f"specialising in {', '.join(spec_parts)}"
                    elif expert:
                        return expert
                    return "in biomedical research"

                persona_context = build_persona_context()

                def build_digest(r):
                    gene  = r["gene"]
                    pname = r.get("protein_name", "N/A")
                    label = f"{gene} ({pname})" if pname and pname != "N/A" else gene
                    lines = [
                        f"Gene: {label} | UniProt: {r.get('uniprot_id','N/A')} | Ensembl: {r.get('ensembl_gene_id','N/A')}",
                        f"  RefSeq summary: {r.get('refseq_summary','N/A')}",
                        f"  Literature synthesis: {r.get('llm_summary_papers','N/A')}",
                        f"  Literature: {r.get('articles_found',0)} abstracts, window {r.get('search_window_used','N/A')}",
                    ]
                    if r.get("references") and r["references"] not in ("N/A",""):
                        lines.append(f"  References:\n{r['references']}")
                    if r.get("pathway_summary") and r["pathway_summary"] != "N/A":
                        lines.append(f"  Pathway summary: {r['pathway_summary']}")
                    if r.get("ot_disease_association") and r["ot_disease_association"] not in ("N/A","Query failed","None found"):
                        lines.append(f"  Disease associations (OT): {r['ot_disease_association']}")
                    if r.get("chembl_molecules") and r["chembl_molecules"] != "None":
                        lines.append(f"  Known drugs (ChEMBL): {r['chembl_molecules']}")
                    if r.get("active_trials") is not None:
                        lines.append(f"  Clinical trials: {r.get('active_trials',0)} active")
                    if r.get("gwas_traits") and r["gwas_traits"] != "None":
                        lines.append(f"  GWAS traits: {r['gwas_traits']}")
                    return "\n".join(lines)

                def make_user_msg(digest_text):
                    return (
                        f"<study_context>\nDisease context: {DISEASE_CONTEXT}\n"
                        f"Organ context: {ORGAN_CONTEXT or 'N/A'}\n"
                        f"Other context: {OTHER_CONTEXT or 'N/A'}\n"
                        f"Full context: {context_desc}\n"
                        f"Genes analysed: {len(successful)}\n"
                        f"Rules: Treat missing fields or toggled-off sources as unavailable evidence, not as negative evidence.\n"
                        f"</study_context>\n\n<per_gene_data>\n{digest_text}\n</per_gene_data>"
                    )

                def run_llm_section(label, system_prompt, user_msg, temperature=0.3):
                    t0 = time.time()
                    try:
                        resp = llm.chat.completions.create(
                            model=LLM_MODEL,
                            messages=[{"role": "system", "content": system_prompt},
                                      {"role": "user",   "content": user_msg}],
                            temperature=temperature, timeout=120)
                        elapsed = time.time() - t0
                        return resp.choices[0].message.content, (resp.usage.total_tokens if resp.usage else 0), elapsed
                    except Exception as e:
                        return f"LLM call failed: {e}", 0, time.time() - t0

                synthesis_results = st.session_state.synthesis_results
                synthesis_tables  = st.session_state.synthesis_tables

                # ── Only call the LLM if synthesis hasn't been run yet ────────
                # After a download re-run, session_state still has the results
                # so we skip straight to the display block below.
                # Determine which sections should run based on toggles + cached state
                should_run = {
                    "9a": RUN_9A,
                    "9b": RUN_9B,
                    "9c": RUN_9C,
                    "9d": RUN_9D,
                }
                already_done = all(
                    not should_run[k] or synthesis_results.get(k)
                    for k in ("9a","9b","9c","9d")
                )

                if not already_done:
                    # ── Only runs once; skipped on re-runs from downloads ─────
                    digest = "\n\n".join(build_digest(r) for r in successful)
                    run_metrics = st.session_state.run_metrics

                    SYS_9A = f"""You are a cell biologist {persona_context}, synthesising patterns across a gene set to provide a high-level mechanistic landscape for a research team.

CONSTRAINTS (apply to every section):
- Use only the provided per-gene data. Do not introduce unsupported genes, pathways, cell types, or causal claims.
- Treat missing or toggled-off fields as unavailable evidence, not as negative evidence.
- Do not repeatedly focus on the same small subset of genes — represent the breadth of the gene set where evidence permits.
- For all bullet points, format each bullet as: **Name**: one sentence describing which genes are involved and their role.
- Aim for concision — this is an orientation document for a research team, not an exhaustive review.
- If a section cannot be supported by the data, write: "Insufficient data to assess."

TASK: Synthesise the per-gene evidence into a mechanistic landscape for the context: {context_desc}. The goal is to identify patterns and themes that orient a research team toward the most meaningful biology, not to exhaustively describe every gene.

OUTPUT FORMAT: Markdown only. Use the headings below in exactly this order. Do not add extra headings, a conclusion, or a preamble.

## Overview
3–5 sentences on the dominant mechanistic patterns emerging from the gene set in the context of {context_desc}. Stay at the theme level — individual genes should only appear here as exemplars of a broader pattern.

## Key biological pathways
Up to 5 bullet points representing the major functional modules across the gene set.

## Upstream drivers vs downstream effectors
Classify genes into exactly three categories. For each, list the relevant genes and explain why they belong there, noting where genes co-operate. Genes with ambiguous roles or insufficient evidence should be noted as such within the most likely category rather than omitted.
**Upstream regulators**: ...
**Signalling intermediates**: ...
**Downstream effectors**: ...

## Cross-gene interactions
Up to 5 bullet points describing direct or indirect functional relationships between specific genes — e.g. one gene regulating another's expression, synergistic activity, or shared complex membership. Weaker or inferred interactions are permitted and should be flagged in Confidence and caveats.

## Confidence and caveats
Based on the synthesised findings above, assess the overall reliability of the synthesis.
- **Supported conclusions**: patterns or findings consistent across multiple genes with direct mechanistic evidence.
- **Cautious inferences**: findings resting on a single gene, indirect evidence, or inferred interactions from Cross-gene interactions.
- **Heterogeneity or gaps**: genes with sparse data, conflicting findings, or not represented in the synthesis — name them explicitly."""

                    if should_run["9a"] and not synthesis_results.get("9a"):
                        with st.spinner("9a — Biological Synthesis..."):
                            text, tok, rt = run_llm_section("9a", SYS_9A, make_user_msg(digest))
                            synthesis_results["9a"] = text
                            run_metrics["9a"] = {"tokens": tok, "runtime": rt}
                    else:
                        text = synthesis_results.get("9a", "")

                    if text and not text.startswith("LLM call failed"):
                        SYS_9A_TABLE = f"""You are a cell biologist {persona_context}, compressing a mechanistic gene set synthesis into a scannable summary table for a research team.

CONSTRAINTS:
- Use only the findings from the provided biological synthesis. Do not introduce new information.
- Aim for concision — each cell should be specific enough to be informative but brief enough to scan.
- If the synthesis cannot support 3 rows of distinct findings, write: "Insufficient data to assess."

TASK: Compress the provided biological synthesis into a single compact markdown table summarising the key mechanistic landscape across the gene set.

OUTPUT FORMAT:
- Output exactly one line before the table: ### Compact Biological Summary
- Then output exactly one markdown table with these columns:
  | Functional module | Representative genes | Mechanistic role | Confidence and caveat |
- 3 to 5 rows. Each row represents one functional module from the synthesis.
- Mechanistic role: one sentence, gene-centred, specific to the context.
- Confidence and caveat: one brief phrase drawn from the Confidence and caveats section of the synthesis.
- Output the title line and table only — no preamble, no commentary, no extra headings."""
                        with st.spinner("9a — Compact table..."):
                            table_text, tok, rt = run_llm_section("9a-table", SYS_9A_TABLE,
                                                            f"<section_text>\n{text}\n</section_text>",
                                                            temperature=0.1)
                            synthesis_tables["9a"] = table_text
                            run_metrics["9a-table"] = {"tokens": tok, "runtime": rt}

                    SYS_9B = f"""You are a drug discovery scientist {persona_context}, assessing the therapeutic landscape of a gene set to orient a research team toward the most actionable opportunities.

CONSTRAINTS (apply to every section):
- Base therapeutic reasoning on the provided drug, clinical trial, disease association, and GWAS data as the primary evidence layer.
- Use the biological synthesis from the previous section — RefSeq summaries, literature findings, and pathway context — as a mechanistic grounding layer to inform and enrich your reasoning, not as primary evidence.
- Do not overstate novelty — a target may be flagged as potentially novel or underexplored only if strong mechanistic or disease association evidence supports it.
- Targets with no evidence across any layer should be noted as insufficient data and excluded from ranking.
- Aim for concision — this is an orientation document for a research team, not an exhaustive review.
- If a section cannot be supported by the data, write: "Insufficient data to assess."

TASK: Produce a therapeutic landscape assessment for the provided gene set in the context of {context_desc}. The goal is to surface the most actionable therapeutic opportunities and gaps across the gene set, grounded in both the therapeutic evidence and the mechanistic context.

OUTPUT FORMAT: Markdown only. Use the headings below in exactly this order. Do not add extra headings, a conclusion, or a preamble.

## Overview
Exactly 4 sentences:
1. The overall therapeutic maturity of this gene set.
2. Which evidence layer is strongest across the set — drug, clinical trial, disease association, or GWAS.
3. The most compelling therapeutic opportunity and why.
4. The most significant gap or unmet need across the gene set.

## Most promising targets
Rank at least 5 but up to 10 targets with available evidence. If fewer than 5 targets have sufficient evidence to rank, include all available and note the limited coverage in the Overview.
For each target, the justification must explicitly reference the evidence it draws from — drug data, clinical trial status, OT association score, or GWAS hit — and be grounded by the mechanistic context where relevant.
Targets with no drug evidence but strong mechanistic or disease association support should be flagged as Potentially novel.
Targets with an existing drug but no clinical trial evidence should be flagged as Potentially underexplored.
Table columns: | Rank | Target | Evidence basis | Justification (≤25 words) | Novelty flag | Caveat (≤15 words) |"""

                    if should_run["9b"] and not synthesis_results.get("9b"):
                        with st.spinner("9b — Therapeutic Landscape..."):
                            text, tok, rt = run_llm_section("9b", SYS_9B, make_user_msg(digest))
                            synthesis_results["9b"] = text
                            run_metrics["9b"] = {"tokens": tok, "runtime": rt}

                    SYS_9C = f"""You are a translational experimental biologist {persona_context}, proposing experiments to validate the most meaningful hypotheses emerging from a gene set analysis to orient a research team toward actionable next steps.

CONSTRAINTS (apply to every section):
- Derive all hypotheses from the provided per-gene data, biological synthesis, and therapeutic landscape as a whole — do not generate hypotheses from general knowledge alone.
- Each hypothesis must be falsifiable and grounded in at least one evidence layer from the input.
- Do not overstate experimental feasibility — flag practical limitations honestly in the caveat.
- Aim for concision — this is an orientation document for a research team, not an exhaustive review.
- If a section cannot be supported by the data, write: "Insufficient data to assess."

TASK: Propose experiments to validate the most promising hypotheses from the gene set in the context of {context_desc}. The goal is to identify the most tractable and informative experiments that a research team could prioritise, grounded in the biological and therapeutic evidence provided.

OUTPUT FORMAT: Markdown only. Use the headings below in exactly this order. Do not add extra headings, a conclusion, or a preamble.

## Overview
Exactly 3 sentences:
1. The overall experimental tractability of this gene set.
2. Which 1–2 targets are the most experimentally accessible and why.
3. The main practical limitation or gap across the gene set.

## Experimental hypotheses
2 to 3 hypotheses, each as a falsifiable statement in the form: "Gene X does Y in the context of Z."
Table columns: | # | Hypothesis | Evidence basis | Key assumption | Caveat |
- Evidence basis: the specific evidence layer(s) this hypothesis is drawn from — per-gene data, biological synthesis, or therapeutic landscape.
- Key assumption: the main unstated premise that must hold for the hypothesis to be testable.
- Caveat: the most significant limitation or risk to testing this hypothesis.

## Proposed experiments
One or more experiments per hypothesis where warranted. The model should determine the most appropriate experiment type — in vitro, in vivo, ex vivo, or patient-derived — based on the hypothesis and the available evidence. Justify the choice of experiment type in the Rationale column.
Table columns: | # | Experiment type | Model system | Perturbation | Primary readout | Rationale for experiment type | Caveat |
- Experiment type: in vitro / in vivo / ex vivo / patient-derived.
- Primary readout: the specific measurable outcome that would confirm or refute the hypothesis — be precise enough that a researcher could design the experiment from this description.
- Rationale for experiment type: one brief phrase explaining why this experiment type is most appropriate for this hypothesis."""

                    prior_9ab = ""
                    if synthesis_tables.get("9a"):
                        prior_9ab += f"\n<biological_summary>\n{synthesis_tables['9a']}\n</biological_summary>"
                    if synthesis_results.get("9b"):
                        prior_9ab += f"\n<therapeutic_landscape>\n{synthesis_results['9b']}\n</therapeutic_landscape>"
                    user_9c = make_user_msg(digest) + (f"\n\n<prior_sections>{prior_9ab}\n</prior_sections>" if prior_9ab else "")

                    if should_run["9c"] and not synthesis_results.get("9c"):
                        with st.spinner("9c — Experimental Proposals..."):
                            text, tok, rt = run_llm_section("9c", SYS_9C, user_9c)
                            synthesis_results["9c"] = text
                            run_metrics["9c"] = {"tokens": tok, "runtime": rt}

                    SYS_9D = f"""You are a drug discovery scientist {persona_context}, synthesising all available evidence across biological, therapeutic, and disease association layers to produce a final integrated target prioritisation for a research team.

CONSTRAINTS (apply to every section):
- Base prioritisation on all provided evidence layers: per-gene data, biological synthesis, therapeutic landscape, disease associations, and GWAS data.
- Apply the following ranking hierarchy per target:
    1. Strength of disease association and GWAS evidence — direct genetic or associative links to the disease context.
    2. Therapeutic evidence — existing drugs, clinical trial status, and OT association scores.
    3. Mechanistic specificity — how well-characterised is the gene's role in the disease pathway from the biological synthesis.
- Overall priority criteria:
    High: strong evidence across all three criteria, with disease association always required.
    Medium: strong evidence in one or two criteria, or moderate evidence across all three.
    Low: evidence in only one criterion, or weak evidence across multiple.
    N/A: no evidence across any layer — exclude from ranking.
- Overall priority values are strictly: High / Medium / Low. No intermediate values.
- Aim for concision — this is an orientation document for a research team, not an exhaustive review.
- If a section cannot be supported by the data, write: "Insufficient data to assess."

TASK: Produce an integrated target prioritisation for the provided gene set in the context of {context_desc}. The goal is to synthesise all evidence layers into a clear, actionable recommendation that a research team can use to focus their efforts.

OUTPUT FORMAT: Markdown only. Use the headings below in exactly this order. Do not add extra headings, a conclusion, or a preamble.

## Overview
Exactly 4 sentences:
1. A clear recommendation on where the research team should focus based on the full evidence across the gene set.
2. The top 1–2 targets and the specific evidence that makes them the strongest candidates.
3. The most significant opportunity or gap that the ranking reveals.
4. One sentence on what would most change this prioritisation if new evidence became available.

## Top target summary
Rank at least 5 but up to 10 targets with available evidence. If fewer than 5 targets have sufficient evidence to rank, include all available and note the limited coverage in the Overview.
Table columns: | Rank | Gene | Protein name | Disease association and GWAS | Therapeutic evidence | Mechanistic specificity | Overall priority |
- Disease association and GWAS: one phrase summarising the strength of genetic or associative evidence.
- Therapeutic evidence: one phrase summarising drug, trial, or OT score evidence.
- Mechanistic specificity: one phrase on how well-characterised the gene's role is in the disease pathway.
- Overall priority: High / Medium / Low only, applied strictly using the ranking criteria above."""

                    prior_9abd = prior_9ab
                    if synthesis_results.get("9c"):
                        prior_9abd += f"\n<experimental_hypotheses>\n{synthesis_results['9c']}\n</experimental_hypotheses>"
                    user_9d = make_user_msg(digest) + (f"\n\n<prior_sections>{prior_9abd}\n</prior_sections>" if prior_9abd else "")

                    if should_run["9d"] and not synthesis_results.get("9d"):
                        with st.spinner("9d — Top Target Summary..."):
                            text, tok, rt = run_llm_section("9d", SYS_9D, user_9d, temperature=0.1)
                            synthesis_results["9d"] = text
                            run_metrics["9d"] = {"tokens": tok, "runtime": rt}

                    # Persist to session_state
                    st.session_state.synthesis_results = synthesis_results
                    st.session_state.synthesis_tables  = synthesis_tables
                    st.session_state.run_metrics       = run_metrics

                # ── Run metrics summary ───────────────────────────────────────
                _metrics = st.session_state.run_metrics
                if _metrics:
                    _metric_rows = []
                    _section_map = [
                        ("core",     "Core pipeline (per-gene LLM)"),
                        ("9a",       "9A — Biological Synthesis"),
                        ("9a-table", "9A — Compact Table"),
                        ("9b",       "9B — Therapeutic Landscape"),
                        ("9c",       "9C — Experimental Proposals"),
                        ("9d",       "9D — Target Prioritisation"),
                    ]
                    total_llm_tok = 0
                    total_llm_rt  = 0.0
                    for key, label in _section_map:
                        m = _metrics.get(key)
                        if m:
                            _metric_rows.append({
                                "Section":  label,
                                "Tokens":   f"{m['tokens']:,}",
                                "Runtime":  fmt_runtime(m["runtime"]),
                            })
                            total_llm_tok += m["tokens"]
                            total_llm_rt  += m["runtime"]
                    if _metric_rows:
                        _metric_rows.append({
                            "Section": "**Total**",
                            "Tokens":  f"**{total_llm_tok:,}**",
                            "Runtime": f"**{fmt_runtime(total_llm_rt)}**",
                        })
                    st.markdown("#### Run Summary")
                    st.dataframe(pd.DataFrame(_metric_rows), use_container_width=True, hide_index=True)
                    st.markdown("---")

                # ── Display (runs every render, cached or freshly generated) ──
                section_labels = [
                    ("9a", "Biological Synthesis"),
                    ("9b", "Therapeutic Landscape"),
                    ("9c", "Experimental Proposals"),
                    ("9d", "Top Target Summary"),
                ]
                for key, title in section_labels:
                    if synthesis_results.get(key):
                        st.markdown(f"---\n### {title}")
                        body = synthesis_results[key]
                        # Strip duplicate leading heading the LLM sometimes outputs
                        body = re.sub(r"(?im)^#{1,3}\s+Top\s+target\s+summary\s*\n", "", body, count=1)
                        body = re.sub(r"(?im)^#{1,3}\s+Therapeutic\s+landscape\s+overview\s*\n", "### Overview\n", body, count=1)
                        st.markdown(body)
                    if synthesis_tables.get(key):
                        st.markdown(synthesis_tables[key])

    # ── Tab 7: Export ─────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Export Results")
        st.caption("All downloads are pre-built — clicking them will not restart the pipeline.")

        # Pull everything from session_state so downloads are always safe
        _results       = st.session_state.results
        _go_bp_data    = st.session_state.go_bp_data
        _synth_results = st.session_state.synthesis_results
        _synth_tables  = st.session_state.synthesis_tables

        df_export = pd.DataFrame(_results)
        EXPORT_COLS = [
            "gene","official_full_name","uniprot_id","protein_name","ensembl_gene_id",
            "refseq_summary","pathway_summary","llm_summary_papers","references",
            "search_window_used","articles_found","chembl_molecules","ot_disease_association",
            "active_trials","trial_phases","gwas_hits","gwas_traits","status"
        ]
        csv_cols = [c for c in EXPORT_COLS if c in df_export.columns]

        # Use a fixed timestamp from session so filenames don't change on re-run
        if "export_ts" not in st.session_state:
            st.session_state.export_ts = datetime.now().strftime("%Y%m%d_%H%M")
        ts       = st.session_state.export_ts
        safe_ctx = re.sub(r"[^\w\-]", "_", DISEASE_CONTEXT or "no_context")
        prefix   = f"{ts}_{ANALYSIS_NAME}_{safe_ctx}"

        # ── Build Markdown once ───────────────────────────────────────────────
        _metrics = st.session_state.run_metrics
        _metrics_md = ""
        if _metrics:
            _section_map = [
                ("core",     "Core pipeline (per-gene LLM)"),
                ("9a",       "9A — Biological Synthesis"),
                ("9a-table", "9A — Compact Table"),
                ("9b",       "9B — Therapeutic Landscape"),
                ("9c",       "9C — Experimental Proposals"),
                ("9d",       "9D — Target Prioritisation"),
            ]
            _mlines = ["## Run Summary\n", "| Section | Tokens | Runtime |", "| --- | ---: | --- |"]
            total_tok, total_rt = 0, 0.0
            for key, label in _section_map:
                m = _metrics.get(key)
                if m:
                    _mlines.append(f"| {label} | {m['tokens']:,} | {fmt_runtime(m['runtime'])} |")
                    total_tok += m["tokens"]
                    total_rt  += m["runtime"]
            _mlines.append(f"| **Total** | **{total_tok:,}** | **{fmt_runtime(total_rt)}** |")
            _metrics_md = "\n".join(_mlines) + "\n\n---\n"

        _meta = [
            "# Gene Research Report\n",
            f"**Analysis name:** {ANALYSIS_NAME}  ",
            f"**Disease context:** {DISEASE_CONTEXT}  ",
            f"**Organ context:** {ORGAN_CONTEXT or 'N/A'}  ",
            f"**Other context:** {OTHER_CONTEXT or 'N/A'}  ",
            f"**Organism:** {UNIPROT_ORGANISM}  ",
            f"**Genes:** {', '.join(r['gene'] for r in _results if r.get('status')=='OK')}  ",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n\n---\n",
        ] + ([f"\n> **Notes:** {NOTES}\n"] if NOTES and NOTES.strip() else []) + [
            "\n## Core Results\n",
        ]
        for r in _results:
            if r.get("status") != "OK": continue
            pname  = r.get("protein_name","N/A")
            header = r["gene"] + (f" — {pname}" if pname and pname != "N/A" else "")
            _meta.append(f"\n### {header}")
            for k, label in [
                ("official_full_name","Official full name"),
                ("uniprot_id","UniProt accession"),
                ("ensembl_gene_id","Ensembl gene ID"),
                ("refseq_summary","RefSeq summary"),
                ("pathway_summary","Pathway summary"),
                ("llm_summary_papers","Literature synthesis"),
                ("references","References"),
                ("chembl_molecules","Known drugs (ChEMBL)"),
                ("ot_disease_association","Disease associations (OT)"),
                ("active_trials","Active clinical trials"),
                ("gwas_traits","GWAS traits"),
            ]:
                val = r.get(k)
                if val is not None and str(val) not in ("N/A","None",""):
                    _meta.append(f"- **{label}:** {val}")

        for key, title in [("9a","Biological Synthesis"),("9b","Therapeutic Landscape"),
                            ("9c","Experimental Proposals"),("9d","Top Target Summary")]:
            if _synth_results.get(key):
                body = _synth_results[key]
                body = re.sub(r"(?im)^#{1,3}\s+Top\s+target\s+summary\s*\n", "", body, count=1)
                body = re.sub(r"(?im)^#{1,3}\s+Therapeutic\s+landscape\s+overview\s*\n", "### Overview\n", body, count=1)
                _meta.append(f"\n---\n\n## {title}\n\n{body}")
            if _synth_tables.get(key):
                _meta.append(_synth_tables[key])

        md_content = _metrics_md + "\n".join(_meta)

        # ── Build HTML from Markdown ──────────────────────────────────────────
        try:
            import markdown as md_lib
            html_body = md_lib.markdown(
                md_content,
                extensions=["tables", "fenced_code", "toc"]
            )
        except ImportError:
            # Fallback: wrap plain text if markdown package missing
            html_body = f"<pre>{html_lib.escape(md_content)}</pre>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GOAT — Gene Research Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 960px; margin: 40px auto; padding: 0 24px;
      color: #1a1a2e; line-height: 1.7; background: #fafafa;
    }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 12px; }}
    h2 {{ color: #2980b9; border-bottom: 1px solid #bdc3c7; padding-bottom: 6px; margin-top: 40px; }}
    h3 {{ color: #27ae60; margin-top: 28px; }}
    table {{
      border-collapse: collapse; width: 100%; margin: 16px 0;
      font-size: 0.9em; box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }}
    th {{
      background: #2980b9; color: white; padding: 10px 14px;
      text-align: left; font-weight: 600;
    }}
    td {{ padding: 8px 14px; border-bottom: 1px solid #ecf0f1; }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    tr:hover {{ background: #eaf4fb; }}
    code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.88em; }}
    pre {{ background: #f0f0f0; padding: 14px; border-radius: 6px; overflow-x: auto; }}
    ul {{ padding-left: 24px; }}
    li {{ margin-bottom: 4px; }}
    hr {{ border: none; border-top: 1px solid #bdc3c7; margin: 32px 0; }}
    .toc {{ background: #eaf4fb; border-left: 4px solid #3498db;
             padding: 16px 24px; border-radius: 4px; margin-bottom: 32px; }}
  </style>
</head>
<body>
{html_body}
</body>
</html>"""

        # ── Download buttons (all pre-built, no re-run triggered) ─────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            csv_buf = StringIO()
            df_export[csv_cols].to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️ Main CSV",
                data=csv_buf.getvalue(),
                file_name=f"{prefix}_gene_research.csv",
                mime="text/csv",
                key="dl_csv",
            )

        with col2:
            if _go_bp_data:
                go_buf = StringIO()
                pd.DataFrame(_go_bp_data).to_csv(go_buf, index=False)
                st.download_button(
                    label="⬇️ GO terms CSV",
                    data=go_buf.getvalue(),
                    file_name=f"{prefix}_go_bp_terms.csv",
                    mime="text/csv",
                    key="dl_go_csv",
                )

        with col3:
            st.download_button(
                label="⬇️ Markdown report",
                data=md_content,
                file_name=f"{prefix}_research_report.md",
                mime="text/markdown",
                key="dl_md",
            )

        with col4:
            st.download_button(
                label="⬇️ HTML report",
                data=html_content,
                file_name=f"{prefix}_research_report.html",
                mime="text/html",
                key="dl_html",
            )

        st.info("💡 The HTML report can be opened directly in any browser and shared with colleagues who don't have Python installed.")

else:
    # Landing state — shown before run
    st.info("👈 Fill in the configuration in the sidebar, then click **▶ Run GOAT** to start the pipeline.")
    st.markdown("""
    **What GOAT does, step by step:**
    1. **NCBI Gene** — resolves each gene symbol to its official name, RefSeq summary, and Ensembl ID
    2. **UniProt** — fetches protein info, GO Biological Process terms, and Reactome pathways
    3. **PubMed + iCite** — searches for relevant abstracts and ranks them by citation count
    4. **LLM synthesis** — writes a cited literature summary per gene
    5. *(Optional)* **OpenTargets** — disease association scores
    6. *(Optional)* **ChEMBL** — known drug interactions
    7. *(Optional)* **ClinicalTrials.gov** — active clinical trials
    8. *(Optional)* **GWAS Catalog** — genetic association evidence
    9. **Cross-gene LLM synthesis** — biological, therapeutic, experimental, and target prioritisation reports
    10. **Export** — CSV, Markdown, and HTML report downloads
    """)
