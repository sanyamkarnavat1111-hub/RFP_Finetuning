EA_prompt = """

### Instruction: 
The following text represents content from Enterprise Architecture (EA) documentation. This content contains established architectural standards, policies, technical constraints, governance requirements, and compliance frameworks that govern all proposed solutions within the enterprise environment.
This text establishes the mandatory architectural standards, technical baselines, interoperability requirements, security policies, and governance constraints that must be satisfied by all proposed solutions. The content defines the permissible technical approaches, approved technology standards, reference architectures, and compliance requirements that form the foundational constraints within which all solutions must operate.
This text constitutes the authoritative source of enterprise-wide architectural standards, policies, and technical constraints that govern solution development and implementation.

### Input:
{text_chunk}

### Output :
No need to provide any , ouput just understand the provided text carefully against the instruction given
"""


proposal_prompt = """
### Instruction: 
The following text represents content from a vendor Proposal document. This content contains the vendor's proposed solution, technical approach, resource commitments, compliance demonstrations, and response to specified requirements.
This text constitutes the vendor's formal response and proposed solution to the requirements outlined in the corresponding RFP. The content includes the vendor's technical approach, methodology, staffing plan, compliance strategy, and delivery commitments that demonstrate how the vendor intends to satisfy the stated requirements.
This text represents the complete expression of the vendor's understanding of the requirements and their proposed method of fulfilling those requirements.

### Input:
{text_chunk}

### Output :
No need to provide any , ouput just understand the provided text carefully against the instruction given

"""


rfp_prompt = """
### Instruction: 
The following text represents content from a Request for Proposal (RFP) document. This content contains procurement requirements, technical specifications, compliance obligations, evaluation criteria, and contractual constraints that define the scope of work and vendor obligations.
This text contains explicit requirements, specifications, and compliance standards that must be understood as the foundational requirements for any response or solution developed in response to this RFP. The content includes mandatory requirements, technical specifications, deliverable expectations, evaluation criteria, and procedural requirements that govern vendor participation and proposal evaluation.
This text represents the authoritative source of requirements and compliance expectations for the specified procurement.


### Input:
{text_chunk}

### Output :
No need to provide any , ouput just understand the provided text carefully against the instruction given
"""