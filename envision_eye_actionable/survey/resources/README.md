# Survey resources

- `schemas/dataset_description.schema.json` and
  `schemas/dataset_structure_description.schema.json` are copies of the CMDS
  v0.1.1 schemas from [AI-READI/cmds-docs](https://github.com/AI-READI/cmds-docs)
  (`static/schemas/v0.1.1/`). Their validation rules are the same as CDS
  v0.1.1 and schema.aireadi.org v0.1.0. The survey uses them to validate the
  JSON it emits.

  License of the two schema files: Copyright (c) 2024 AI-READI Consortium.
  Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
  Copied unmodified from AI-READI/cmds-docs `static/schemas/v0.1.1/`. The
  cmds-docs repository carries no license file of its own (checked
  2026-09-24); the license is the one of the CMDS/CDS specification,
  AI-READI/cds-specification (`LICENSE`: Attribution 4.0 International,
  Copyright (c) 2024, AI-READI Consortium). They are NOT covered by this
  repository's MIT license. The same notice is in `schemas/LICENSE.md`,
  which ships with every wheel and sdist next to the schema files.
- `dicom_mapping.json` maps the classifier classes to DICOM attributes
  (Modality, SOP Class, Image Type, anatomy and device codes). It also holds
  the pixel-header derivation rules, the laterality regex and the
  manufacturer hints. It was checked against DICOM PS3 2026d on
  dicom.nema.org on 2026-09-24, and its sources are listed in its `_meta`
  block.
