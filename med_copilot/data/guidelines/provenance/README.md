# Guideline provenance manifest

Verified on **2026-10-22** against the project's actual `documents.parquet` and local source text, official NICE overview pages and guidance PDFs, NCBI PubMed citation XML, and publisher-deposited Crossref DOI metadata.

## Files

- `../guideline_manifest.csv`: one flat record per indexed document, suitable for analysis and review.
- `../guideline_manifest.json`: the complete manifest, including all 226 document IDs, bibliographic metadata, field verification states, notes, and unresolved fields.
- `verification_evidence.json`: document-level source URLs, retrieval timestamps where captured, response hashes, bibliographic metadata, local ISBN checks, and local update-history date labels and character offsets.
- `verification_summary.json`: counts and exceptions derived from the manifest.
- `validate_manifest.py`: offline validation against the document table, local inputs, evidence, and CSV.

## Coverage and verified facts

The manifest covers exactly **226 documents**: **221 NICE NG-series extracts** and **five journal guidelines**. Every record joins to the existing long `document_id` and `document_numeric_id`. Every local text was compared with its indexed document text, and every document ID was independently reproduced as the SHA-512 of that text.

The original publication identity, title, issuing organizations, publication date, and source URL are verified for every record. All 221 NICE overview pages were retrieved successfully. NICE briefly returned a service-interruption page during research, but subsequent requests succeeded for every overview and for 220 official guidance PDFs. NG51's overview no longer supplies a guidance PDF.

The five journal guidelines were verified by matching the DOI in the local text to both NCBI PubMed XML and Crossref. Their complete indexed article titles also match the local text after whitespace, Unicode, and punctuation normalization. Journal issue dates and electronic publication dates are separate fields. Month-only issue dates are retained as `YYYY-MM`; no day has been invented.

Joint issuing organizations are retained: CHEST/ATS for ventilator liberation, BTS/NICE/SIGN for NG244 and NG245, and SCCM/ESICM for the Surviving Sepsis Campaign. Crossref's publisher field is labeled `publisher_as_reported_by_crossref`, because current DOI metadata can reflect a publisher transfer and does not establish the original publisher.

## Version limitations

**Publication identity verification does not establish the exact historical PDF snapshot or text revision.**

For the 221 NICE records:

| `version_verification_status` | Count | Meaning |
| --- | ---: | --- |
| `isbn_matches_current_pdf_exact_revision_unverified` | 203 | Every local ISBN matches the current official PDF. ISBNs can persist through revisions; the exact local revision date remains unverified. |
| `historical_isbn_differs_from_current_pdf_revision_unverified` | 17 | The local and current official PDF ISBNs differ. The original title and publication identity are verified, but the current PDF is a different edition and cannot verify the historical local revision date. |
| `superseded_guideline_pdf_unavailable_revision_unverified` | 1 | NG51 is officially superseded by NG253, NG254, and NG255. Its identity and bibliographic dates are verified from its overview, but that overview no longer provides its historical PDF. |

The 17 ISBN differences are **NG12, NG23, NG24, NG28, NG49, NG59, NG88, NG126, NG136, NG194, NG195, NG198, NG217, NG220, NG229, NG235, and NG246**.

The five journal records have `doi_and_publication_edition_verified`. Their DOI identifies the journal publication/edition; this does not verify the original downloaded PDF binary or every character of the extracted text.

All 226 records explicitly set `exact_original_pdf_snapshot_verified` to `false`. The original acquisition URL, original acquisition date, and original PDF binary were not retained in the available artifacts. The current source URL is not presented as proof of the original download history.

## Date and title semantics

- `publication_date` is the original NICE publication date or the journal issue date, as specified by `publication_date_kind`. It is not the local extraction date.
- `online_publication_date` is NCBI's explicitly supplied electronic article date. For the ATS ARDS guideline, NCBI supplies no electronic article date. Crossref reports `2024-01-01` as its online date, but the original first-online date was not independently established and is explicitly flagged as unverified.
- `official_last_updated_date` is the date currently labeled “Last updated” by NICE. A blank means the overview does not list that field. It does not prove that no minor changes occurred.
- `local_latest_documented_change_month` is the latest month/year label found at the start of an entry in the local extract's **Update information** section. It is available for 180 documents. It establishes that a change is mentioned in the local text; it does not establish the exact revision or download date. Missing values are labeled `not_found_in_local_update_history`.
- `index_creation_timestamp` is the historical filesystem timestamp stored by the indexer. It is not a publication, edition, or source-download date.
- `original_title` preserves the title belonging to the indexed publication. `current_official_title` separately records today's NICE title. NG36, NG49, and NG52 retain historical title wording verified against indexed official NICE pages and the local extracts.
- `extract_page_range_from_filename` records the page range claimed by the local filename. It is not a verified page range in the current, potentially repaginated PDF and is not a chunk-level citation.

## Inspecting the evidence

Start with a manifest `document_id`, find the same ID in `verification_evidence.json`, and inspect its official source URLs and checks. NICE evidence records retain the overview's publication/update labels, current PDF ISBNs, and hashes of the retrieved overview and PDF. Local update-history offsets refer to the exact indexed `documents.parquet` text using Python string character offsets.

For the journal records, the evidence retains the NCBI request URL and response hash and the relevant publisher-deposited Crossref metadata. The PubMed HTML website returned a cookie challenge; the official NCBI citation API successfully returned all five records, so the verification does not rely on those challenged HTML responses. SCCM's official Surviving Sepsis Campaign page establishes its joint SCCM/ESICM organization identity.

Downloaded official pages/PDFs and full API responses were kept in a temporary research cache, not added to the repository. The committed evidence retains bibliographic facts, source addresses, and response fingerprints. These fingerprints identify the responses inspected; they are not archival copies, and future live responses may change.

This work does not reconstruct section titles or character spans for guideline recommendations, certify clinical currency, reconcile the paper's complete corpus, rebuild an index, or change retrieval/generation behavior.

## Offline validation

From the repository root, using `medcopilot` or another environment with pandas and Parquet support:

```bash
python med_copilot/data/guidelines/provenance/validate_manifest.py
```

The validator checks exact document coverage, IDs, source text hashes, file mappings, required verified fields, dates, ISBN checksums, evidence coverage, uncertainty flags, and CSV/JSON consistency. It makes no network requests and does not establish that a source is still unchanged or available after the verification date.
