# Contract & Policy PDFs — Download Instructions

This project uses **real public-domain procurement documents** as the unstructured
knowledge base. These are freely available and contain genuine contractual language
(penalty clauses, payment terms, termination conditions, force majeure, etc.).

---

## Documents to Download

### 1. UNOPS Standard Contract for Services
**URL:** https://www.unops.org/SiteCollectionDocuments/Procurement/UNOPS-contract-templates.zip
**Direct PDF:** https://www.unops.org/SiteCollectionDocuments/Procurement/Long-form-services-contract.pdf

- Save as: `data/contracts/vendor_contract_unops_services.pdf`

### 2. World Bank Standard Procurement Document — Consultants
**URL:** https://www.worldbank.org/en/projects-operations/products-and-services/brief/procurement-new-framework
**Direct:** https://thedocs.worldbank.org/en/doc/standard-form-contracts

- Save as: `data/contracts/vendor_contract_worldbank_consulting.pdf`

### 3. UN Procurement Manual (Policy Document)
**URL:** https://www.un.org/Depts/ptd/sites/www.un.org.Depts.ptd/files/files/attachment/page/pdf/UN_Procurement_Manual_2021.pdf

- Save as: `data/policies/un_procurement_manual.pdf`

### 4. World Bank Procurement Regulations
**URL:** https://documents.worldbank.org/en/publication/documents-reports/documentdetail/178331525702868310/world-bank-procurement-regulations-for-ipf-borrowers

- Save as: `data/policies/worldbank_procurement_regulations.pdf`

---

## Alternative — Use Sample PDFs for Testing

If you want to start immediately without downloading, run this script to
generate minimal placeholder PDFs for pipeline testing:

```bash
python data/contracts/create_sample_contracts.py
```

This creates 3 synthetic contract PDFs with realistic clause language
so you can run the full ingestion pipeline and test retrieval.

---

## Folder Structure After Download

```
data/
├── contracts/
│   ├── vendor_contract_unops_services.pdf
│   ├── vendor_contract_worldbank_consulting.pdf
│   └── (add more vendor contracts here)
├── policies/
│   ├── un_procurement_manual.pdf
│   └── worldbank_procurement_regulations.pdf
```

---

## CV / Interview Framing

When presenting this project:

> "The unstructured knowledge base uses real public procurement contract
> templates from UNOPS and the World Bank, combined with the UN Procurement
> Manual and World Bank Procurement Regulations — all freely available
> public-domain documents. This means the RAG agent answers genuine
> procurement questions using real contractual language, not toy examples.
> This mirrors how consulting firms build presales demos before a client
> contract is signed — they use public templates because they cannot use
> client data."
