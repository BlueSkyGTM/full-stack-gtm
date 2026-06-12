THE 80/20 GTM ENGINEERING PLAYBOOK
Fullstack GTM Engineering — Core Curriculum Reference
This document is the practitioner core of the Fullstack GTM Engineering curriculum. It covers the
foundational systems, methods, and tools that constitute modern GTM engineering in 2025–2026.
The term 80/20 does not mean abbreviated. It means this is the work that actually moves pipeline —
the essential machinery of outbound, enrichment, signals, and multichannel execution that most
GTM teams either do manually, do partially, or do not do at all.
The document is organized in four sections: TAM and outbound foundation, multichannel and
signal-based execution, scraping and real-time signal detection, and the enterprise GTM layer. Each
section contains expanded definitions, the underlying logic, the tool stack, and the key performance
indicators a GTM engineer is responsible for.
Source material: The 80/20 GTM Engineer Handbook by Michael Saruggia (Growth Lead LLC),
The Enterprise GTM Engineer Playbook, Clay University, Apollo Academy, HubSpot Academy,
Salesforce Trailhead, Made With ML (Anyscale), ZoomInfo, Demandbase, 6sense, Gong,
Smartlead, n8n, Make, and a synthesis of 2026 GTM engineering job market data.
TABLE OF CONTENTS
Section 1 — TAM & Outbound Foundation
1.1 TAM Mapping
1.2 TAM Refinement & ICP Scoring
1.3 Copywriting & Testing
1.4 Deliverability & Cold Email Infrastructure
Section 2 — Multichannel & Signal-Based Outbound
2.1 LinkedIn Automation & ABM
2.2 Cold Calling Infrastructure
2.3 Micro Lists & Outbound 3.0
Section 3 — Scraping & Signal Detection

3.1 Scraping Directories & Niche Sources
3.2 News-Led & Signal-Based Outbound
3.3 Inbound-Led Outbound
Section 4 — Enterprise GTM Layer
4.1 PLG Playbook
4.2 CRM Playbook
4.3 Job Change Playbook
4.4 Social Signals Playbook
4.5 Conference & Event Playbook
4.6 Ex-Champions Playbook
4.7 Outbound at Scale
4.8 GTM Analysis & Premium Positioning

SECTION 1 — TAM & OUTBOUND FOUNDATION
The outbound foundation is where every GTM engineering engagement begins. Before anything
else — before copy, before sequences, before personalization — you need a list that actually reflects
your market. Most outbound fails not because of bad copy but because of a bad list sent with generic
positioning. The TAM process is the discipline of building a list worth sending to.
1.1 TAM Mapping
TAM mapping is the first step of every outbound campaign and, done correctly, the hardest. The
task is to source the entire addressable market in a broad way before any filtering occurs. The
fundamental question to answer is: where do these businesses exist on the internet? The answer
changes by industry and by company size, and getting it right is a strategic decision, not a
mechanical one.
For most B2B companies, LinkedIn is the default source. Clay's Find Companies and Find People
features scrape LinkedIn directly, making it the primary source for any market where companies are
incentivized to maintain a professional presence. This covers the majority of mid-market and
enterprise B2B targets. However, LinkedIn fails consistently for certain categories: small
e-commerce businesses under $1M in revenue have little incentive to be on a B2B platform; local
businesses such as restaurants, gyms, dental practices, and retailers live on Google Maps, not
LinkedIn; and niche industry directories often contain better coverage than any general platform.
The practitioner rule is simple: identify the one or two sources where the density of your target
company type is highest, and build from there. For e-commerce, Storeleads provides coverage that
LinkedIn cannot. For local businesses, Google Maps scraped via the SerperDev API yields name,
address, phone, and email at scale. For niche markets — game development studios, real estate
brokers, Amazon sellers, law firms — there are industry directories (gamedevmaos.com,
Realtors.com, Smart Scout, Martindale) that aggregate exactly the companies you need. For 50% of
markets, LinkedIn is sufficient. For the other 50%, you need to find the niche provider.
STACK
– Clay (Find Companies, Find People) — primary LinkedIn source for B2B markets
– Storeleads — e-commerce businesses by platform, revenue tier, and technology stack
– SerperDev API — Google Maps scraping for local businesses at scale via Clay RSS feed
– Apollo Export — people-at-company lookup when Clay's free tier coverage is insufficient

– Niche directories — scraped via Claygent or Apify depending on site complexity
KPIS
– TAM size (number of companies in scope)
– SAM conversion rate (percentage that pass ICP filters after refinement)
– Cost per lead sourced
1.2 TAM Refinement & ICP Scoring
A broad TAM list is not a usable list. After the initial pull, the list contains companies that
technically match the surface-level criteria but fail on the details that determine whether they can
actually buy. An e-commerce list will include service businesses that happen to use Shopify. A
LinkedIn search for marketing agencies will include freelancers, holding companies, and adjacent
firms that are not actual prospects. TAM refinement is the process of filtering this dirty list down to
the companies that genuinely reflect your ICP.
The refinement is done using a combination of four tools applied in sequence: GPT via the OpenAI
API for open-ended classification (does this company sell physical products or services?), Claygent
for web research on companies that cannot be classified from structured data alone, Clay Formulas
for boolean logic and column construction, and Lead Magic for email enrichment and validation.
The output of this process is a single ICP column that aggregates all the filter answers into a binary:
in or out. Every company that passes this column is a valid prospect. Every company that fails is
excluded from the campaign.
The people-finding step follows refinement. Once you have a clean company list, you identify
decision-makers using a waterfall: Clay's native Find People At Company (free, but limited to open
LinkedIn profiles), Claygent for web scraping when LinkedIn profiles are private, and Apollo
Export as the highest-coverage option for large organizations where decision-maker data needs to be
comprehensive. For smaller companies, the info@ email and phone number scraped directly from
the company website via Claygent is often the most effective contact point, since owners read those
inboxes directly.
ENRICHMENT TOOLS IN THE REFINEMENT STACK
– GPT via OpenAI API — text-based classification: product vs service, industry verification,
keyword detection

– Claygent — open-ended web research: technology confirmation, company description analysis,
phone and email scraping
– Lead Magic — B2B email enrichment, email validation, company and person enrichment via
LinkedIn
– Clay Functions (free) — boolean logic, column formulas, conditional filtering without API credits
– Apollo Export — highest-coverage people-at-company data for large-organization prospecting
ICP COLUMN LOGIC
The ICP column should aggregate all filter results into a single true/false field. A typical
construction combines: industry match (GPT classification), size match (employee count or revenue
range from enrichment), technology match (tech stack from Clay or HG Insights), and exclusion
flags (existing clients, competitors, industries you cannot serve). When all conditions return true, the
record is ICP. This column then controls downstream routing — only ICP-true records enter the
sequence.
KPIS
– ICP conversion rate (percentage of TAM that passes refinement)
– Email validity rate (target: 95%+ valid or safe-to-send)
– Cost per enriched, validated contact
1.3 Copywriting & Testing
Most conversations about cold email copywriting miss the point. The debate over subject lines,
CTAs, and email length is secondary to the single question that determines whether a campaign
works or not: is the value proposition resonant with this list? Everything else — tone,
personalization, follow-up cadence — is optimization around a core that either works or does not.
The 80/20 rule applied to cold email testing is this: test only the value proposition and the list. Not
the subject line. Not the call to action. Not the level of personalization. These variables matter, but
they matter after you have found a value proposition that generates at least one interested reply for
every 250 emails sent. Before that threshold, changing the subject line or adding a personalized first
line will not move the needle. It will just create noise in your data. Send 500 emails per value
proposition variant to get a statistically meaningful result. Launch three to six value propositions
simultaneously. When one crosses the 1-in-250 threshold, you have a winner. Only then do you
begin testing secondary variables.

The template that consistently outperforms more complex structures is deliberately simple: a
personalized first line that demonstrates you researched this specific person or company, a single
sentence stating the outcome you can generate (not features, not process — outcome), a one-phrase
description of the unique system or method that makes the outcome credible, a social proof
reference, and a soft close asking whether they are interested in speaking. The copy fits under 100
words. It contains no links, no images, no HTML formatting. It reads like it was written by a human,
because that is what passes spam filters and what earns replies.
THE CORE TEMPLATE
Hi {first name}, {personalized first line}. If I could {generate X result} using {unique system}, {social
proof} — would you be interested in speaking? — [name]
AI PERSONALIZATION FRAMEWORKS
The personalized first line is where AI adds the most value in cold email. Five approaches produce
the highest quality first lines without requiring excessive credits or build time:
– Mention how the list was built — acknowledging a specific signal that triggered the outreach (saw
your Shopify store sells X, noticed you recently opened in Y location)
– Notice a problem they could have — a question-based observation about a gap or inefficiency
visible from their public presence
– Reference what a similar competitor is doing — social proof by proximity (companies like X and
Y in your space are scaling using this approach)
– Name something they do not have — a gap-based hook identifying an obvious missing element
(noticed you do not have an outbound team)
– Geographic proximity signal — for local and regional businesses, referencing their city,
neighborhood, or a local landmark creates relevance
TESTING PROTOCOL
– Send 500 emails per value proposition variant before evaluating results
– Threshold for a working value proposition: 1 interested reply per 250 emails sent
– Good performance: 1 per 150. Winner: 1 per 100 or better
– Only after finding the winning value proposition: begin A/B testing subject line, CTA,
personalization depth, follow-up copy
KPIS
– Reply rate (total replies ÷ emails sent)

– Interested rate (interested replies ÷ emails sent) — the primary metric
– Value proposition benchmark: 1 interested per 250 minimum, 1 per 100 target
1.4 Deliverability & Cold Email Infrastructure
Email deliverability is the infrastructure layer that determines whether your campaigns actually
reach inboxes. Companies lose 30 to 50% of their outbound results not from bad copy or wrong
targeting but from poor inbox management and deliverability setup. This is the most technical
component of the outbound foundation and the most commonly under-resourced.
The domain and mailbox architecture follows a simple rule: one domain supports a maximum of 15
email addresses, and each address starts at three emails per day ramping up to a maximum of 15
over four to six weeks. Warmup runs in parallel with sending — 25 warmup emails per day per
address at a 60 to 70% reply rate throughout the life of the domain. The sending domain is always a
secondary domain, never the primary business domain. If a domain gets flagged, the primary
domain reputation is protected.
Email validation is non-negotiable before any send. Lead Magic classifies emails into four
categories: valid with empty substatus (35% of lists — bounce rate under 2%, send freely), valid
catch-all (25% — bounce rate 2–9%, send with caution), invalid catch-all (20% — too risky), and
invalid (20% — do not send). Targeting only the valid and valid catch-all segments eliminates the
majority of bounce risk. The benchmark is bounce rate under 2–3% on any campaign.
Inbox management is where most GTM engineers underinvest. Smartlead provides a master inbox
that aggregates all replies across all sending addresses. One SDR or AE can manage ten or more
client inboxes from this single interface by tagging replies: INTERESTED, NOT INTERESTED,
OUT OF OFFICE. Interested replies trigger a cold call immediately — the reply rate from this
protocol significantly exceeds email-only follow-up. A HubSpot integration creates a new contact in
the outbound pipeline automatically when a reply is tagged as interested. For prospects who reply
and then go silent, a calendar invite sent three days later ("I put some time on your calendar for X,
does this work?") recovers a meaningful percentage of ghosts.
HARD RULES
– No links anywhere in the email body or signature
– No images, no HTML formatting, no video embeds
– Email body under 100 words for initial send

– No unsubscribe link in the first touch (adds it to the marketing email mental category)
– Signature: first name and last name only — no title, no company, no phone in the first message
– No click tracking. Open tracking is acceptable.
– One domain maximum 15 addresses. Warmup 25 per day at 60–70% reply rate throughout.
STACK
– Smartlead — sending platform, warmup engine, master inbox, HubSpot integration
– Lead Magic — email validation (valid/catch-all/invalid classification), B2B enrichment
– Secondary domains — purchased specifically for outbound, kept separate from primary domain
KPIS
– Bounce rate (target: under 2–3%)
– Inbox placement rate
– Interested reply rate per campaign
– Contactability rate for cold calling follow-up

SECTION 2 — MULTICHANNEL & SIGNAL-BASED OUTBOUND
Email alone is no longer sufficient for serious outbound. The 2025–2026 landscape requires a
multichannel approach — not because email is dead, but because email combined with LinkedIn
and cold calling produces meeting rates that are two to three times higher than any single channel in
isolation. The question is not whether to add channels but how to sequence them intelligently so
each touch builds on the last.
2.1 LinkedIn Automation & ABM
LinkedIn is the highest-signal channel in outbound — not because it gets the most replies, but
because a LinkedIn message from a credible profile to an active user carries more weight than a cold
email. The problem is scale. LinkedIn's connection request limits are strict, and most accounts send
to people who are not active on the platform, wasting the few weekly touchpoints available. The
GTM engineering fix is to make LinkedIn a sniper channel, not a spray channel.
HeyReach is the primary tool for safe LinkedIn automation at scale. It handles connection requests,
messages, profile visits, likes, and in-mails across multiple accounts simultaneously, making it
suitable for agencies running outbound for multiple clients and for enterprise teams with multiple
SDRs. The safety record is strong when used within reasonable volume limits.
The most effective targeting filter on LinkedIn is connection count. Filtering for profiles with 500 or
more connections identifies people who have intentionally built their network — they are active on
the platform and they read their messages. A prospect with 50 connections accepted the default
LinkedIn settings; a prospect with 500 made a deliberate choice to engage with the platform. Reply
rates on LinkedIn outreach improve dramatically when this filter is applied, and the cost per
response drops significantly because you are not wasting connection requests on accounts that never
log in.
The highest-performing LinkedIn campaigns use content as the hook rather than a direct pitch. The
TOP 100 ECOMM LIST case study demonstrates this: a market research project targeting
e-commerce managers, framed as a publication honoring the top professionals in the field, generated
a 56% reply rate and 75% positive replies. The outreach was not a sales pitch — it was an invitation
to be featured. The LinkedIn message served as the anchor; email follow-up was only triggered if
the prospect did not reply on LinkedIn. This approach only works when the TAM is small (under
2,000 companies), the ACV is above $30K, and the content hook is genuinely compelling to the
target persona.

The executive outreach campaign extends this logic by leveraging the LinkedIn networks of
company executives. An executive with 2,000 connections has relationships that a junior SDR does
not. HeyReach connects safely to executive accounts and sends personalized sequences to their
first-degree connections. Vidyard enables one-to-many personalized videos where the executive
introduces themselves by the prospect's name — a format that produces significantly higher
response rates than text alone. One SDR manages all inboxes generated by the executive accounts
and books meetings while positioning themselves as working directly with the executive.
CREATIVE PLAY FORMATS
– Market research project — invite prospects to participate in a publication or ranking (ego-based,
opt-in framing)
– Podcast invite — offer to feature the prospect as a guest, positioning them as a thought leader
– Micro offline event — invite to a small dinner, breakfast, or in-person session with peers
– Micro online event — invite to a virtual roundtable with other decision-makers in their space
– Company visit — offer to come to their office for an in-person assessment or audit
STACK
– HeyReach — LinkedIn automation, multi-account management, sequence builder
– Clay — enrichment and filtering (500+ connections filter, active user identification)
– Vidyard — 1-to-many personalized video for executive outreach sequences
– Smartlead — email anchor sequences triggered when LinkedIn goes unanswered
KPIS
– LinkedIn connection acceptance rate
– Reply rate per sequence type (direct pitch vs content hook vs research invite)
– Meeting rate from LinkedIn-originated conversations
2.2 Cold Calling Infrastructure
Cold calling remains one of the highest-ROI outbound channels when the deal size and TAM
characteristics justify the investment. The economics work when the average contract value is above
$20,000 to $30,000, because the cost of dialers, mobile phone data, and the SDR time required to
operate a calling program is substantial. Below that ACV threshold, email and LinkedIn are more
cost-efficient per meeting booked.

The dialer selection decision is driven primarily by TAM size and contactability rate. For small
TAMs with high ACV, a CRM-embedded dialer (HubSpot, Close, Salesforce) provides the
high-touch one-to-one experience appropriate for enterprise accounts where knowing the company
context before each call matters. For mid-market volumes where SDR productivity is the primary
constraint, a power dialer (Apollo, PhoneBurner) enables sequential dialing through a curated list
without the friction of a CRM interface. For large TAMs where contactability is low — industries
like healthcare where getting a live person requires dialing many numbers — a predictive dialer
(Nooks, Salesfinity) calls two to five numbers simultaneously and connects the SDR to the first
person who answers, dramatically increasing productive call time per hour.
Phone number sourcing is the most underestimated challenge in cold calling programs. For smaller
companies and local businesses, Claygent can scrape phone numbers directly from company
websites — most small business owners have a phone number in the footer, contact page, or about
page. For larger organizations where personal mobile numbers are required to bypass gatekeepers,
Clay's Mobile Phone Waterfall uses 17 data providers in sequence to find verified mobile numbers.
The cost is approximately $0.20 per mobile number found, which is expensive at scale but justified
when contactability rates without mobile data are near zero.
DIALER SELECTION MATRIX
– CRM dialer (HubSpot, Close) — high ACV ($50K+), small TAM (<500 accounts),
relationship-driven selling
– Power dialer (Apollo, PhoneBurner at $200+/mo) — mid-market ACV, SDR productivity the
priority, omnichannel approach
– Predictive dialer (Nooks, Salesfinity at $300/mo, 3–5 seat minimum) — large TAM, low
contactability industries, high volume programs
KPIS
– Contactability rate (live answers ÷ dials) — low contactability signals a mobile data problem or
dialer mismatch
– Calls per SDR per day — benchmark varies by dialer type; must be tracked religiously
– Conversion rate (meetings booked ÷ live conversations)
2.3 Micro Lists & Outbound 3.0

The outbound landscape has bifurcated. High-volume, high-tech outbound — over one million
emails per month, real-time deliverability analytics, continuously optimized offers — is the domain
of a small number of sophisticated agencies and tech-enabled services. It requires infrastructure that
almost no company can build internally and maintain. For the vast majority of GTM engineers, this
is not the relevant approach.
The second approach, which is the focus of practical GTM engineering in 2025, works at smaller
scales with much higher precision. It is the approach Miles Veth predicted when he described
outbound going back to one-to-one, human-sent email: highly researched, deeply personalized, sent
to micro-lists of prospects carefully selected for maximum relevance. The ACV threshold for this
approach is around $30,000. Below that, the time investment does not pay off. Above it, a
well-executed micro-list campaign can outperform a high-volume blast by an order of magnitude on
a per-meeting basis.
The operational mechanics of micro-list outbound are simple. Clay Sculpt builds the list. The copy
is written using deep account-level research — finding angles like the city a prospect grew up in, a
recent podcast they appeared on, a company announcement that creates relevance. The sends are 15
emails per day from a personal Gmail account and 15 LinkedIn messages per day, managed by a VA
or executed personally. Gift card offers ($100–$400 Amazon cards) can serve as hooks for the
highest-value target accounts where generic outreach would be ignored — the gift frames the
interaction as research-based rather than sales-based.
The micro-list philosophy applies to TAM segmentation as well. Rather than treating a large TAM
as one campaign, a skilled GTM engineer breaks it into sub-segments by vertical, technology stack,
company lifecycle stage, or geography, and writes angle-specific value propositions for each
segment. A B2C company with offline stores has different pain points if it is a food franchise versus
a luxury retailer versus a healthcare chain. Each sub-segment gets its own list, its own angle, and its
own value proposition — all tested and optimized independently.
MICRO-LIST SEGMENTATION EXAMPLE
Starting TAM: 500+ employee B2C companies with offline retail locations. Three sub-segments:
(1) food franchising chains — pain point is new store ramp-up speed; (2) companies with an older
average customer demographic — pain point is digital acquisition without alienating existing base;
(3) companies using Salesforce as CRM — pain point is CRM data quality and enrichment. Each
segment receives a different value proposition, a different personalization angle, and a different lead
magnet or hook.
OPERATIONAL FORMULA

– Clay Sculpt for research and list building
– $100–$400 gift card or research-based hook for highest-value accounts
– One VA or the GTM engineer themselves managing sends
– 15 emails per day from personal Gmail, 15 LinkedIn messages per day
KPIS
– Reply rate per micro-segment
– Meeting rate per micro-segment
– Cost per meeting across approaches (micro-list vs standard outbound)

SECTION 3 — SCRAPING & SIGNAL DETECTION
Signal detection is what separates reactive outbound — sending to companies because they match a
firmographic profile — from proactive outbound that reaches prospects at the exact moment they
have a problem. A company that recently opened a new location, received funding, hired a new
executive, or started running ads for the first time is not just a profile match. It is a company in
motion. Motion is what creates buying intent.
3.1 Scraping Directories & Niche Sources
The single most important skill in TAM mapping is knowing where your target companies actually
exist online. LinkedIn covers the majority of professional B2B markets, but many industries have
better coverage in niche directories, marketplaces, or platforms specific to their sector. A GTM
engineer who can identify and scrape the right source for any niche has a durable data advantage
that competitors using only standard enrichment tools cannot replicate.
The strategic question for every new market is: where does the density of my target company type
exist online? For game development studios, it is gamedevmaos.com. For real estate brokers, it is
Realtors.com. For schools, it is Niche.com. For Amazon sellers, it is Smart Scout. For churches, it is
the Southern Baptist Convention directory. For companies with open jobs, it is Indeed. These are not
obscure resources — they are obvious once you ask the right question. The skill is asking it before
defaulting to LinkedIn.
Claygent is the primary tool for scraping any directory that does not have a dedicated API or Clay
integration. It functions as an AI web scraper that can navigate a website, extract structured data,
and return it in a Clay-compatible format. For sites with aggressive anti-bot protections, ZenRows
provides a bypass layer that handles rotating proxies, JavaScript rendering, and CAPTCHA
avoidance. Apify provides pre-built scrapers for common platforms (LinkedIn, Google Maps,
Instagram, TikTok) when a custom Claygent approach would be slower or more expensive.
NICHE SOURCE EXAMPLES BY INDUSTRY
– E-commerce businesses — Storeleads (filter by platform, revenue, technology stack)
– Local businesses (restaurants, gyms, dentists) — Google Maps via SerperDev API
– Real estate brokers — Realtors.com
– Game development studios — gamedevmaos.com
– Schools and universities — Niche.com

– Amazon businesses — Smart Scout
– Job openings by company — Indeed.com
– Churches and nonprofits — Sector-specific directories (SBC, Guidestar)
STACK
– Clay + Claygent — primary scraping layer for any web-accessible directory
– ZenRows — anti-bot bypass for protected sites when Claygent is blocked
– Apify — pre-built scrapers for common platforms with higher volume requirements
– SerperDev API — Google Maps local business data at scale
3.2 News-Led & Signal-Based Outbound
News-led outbound is one of the most powerful GTM engineering techniques available because it
solves the timing problem. A company is always in your TAM, but it is only in the market —
actively looking for solutions — at specific moments. Funding events, executive hires, new product
launches, regulatory changes, and physical expansion all create buying windows. A GTM engineer
who can detect these moments in real time and route the right prospect to the sales team
immediately has a structural advantage over competitors who are still sending to static lists.
Google News is the most accessible real-time signal source. By setting up a keyword monitoring
system — RSS feeds from Google News piped into Clay — a GTM engineer can surface relevant
events at scale across their entire TAM. The workflow is: identify keywords that predict buying
intent for your product ("new restaurant opening," "acquired" + industry, "raises Series B," "hires
VP Sales"), pipe those keywords into a Google News RSS feed, route the results to a Clay table, use
Claygent to extract the company name and domain from the article, enrich with decision-maker
data, score for ICP fit, and route qualified matches to a sales sequence — all automated, all
real-time.
Active advertising is a second high-value signal. Meta, Google, and LinkedIn all maintain public ad
libraries that are freely accessible. A company running ads is a company that has a growth budget
and is willing to spend on acquisition. The type of ad reveals more: a lead generation ad ("get a free
demo") indicates they are actively building pipeline; an awareness ad indicates they are in an earlier
stage. Clay workflows that scrape these libraries and filter by spend level, ad format, and recency
identify companies in growth mode before any direct competitor outreach reaches them.

Job postings are a third signal layer. A company hiring a VP of Sales is building an outbound
function. A company hiring data engineers is building data infrastructure. A company posting five
SDR roles simultaneously is scaling its outbound program and likely evaluating tools to support it.
Indeed, LinkedIn Jobs, and Greenhouse all provide public job data that can be scraped and
incorporated into a signal scoring system in Clay.
NEWS SIGNAL WORKFLOW
– Define keywords that predict buying intent for your product category
– Set up Google News RSS feeds for each keyword set
– Route RSS feed results to a Clay table via webhook
– Use Claygent to extract company name and domain from article text
– Enrich company and decision-maker data using the standard waterfall
– Score for ICP fit using the ICP column from Section 1
– Route ICP-qualified matches to automated outbound sequence with the news event as context
ADS LIBRARY SIGNAL WORKFLOW
– Identify the relevant ads library for your target market (Meta for consumer-adjacent B2B,
LinkedIn for enterprise)
– Build a Clay table that scrapes the library on a scheduled basis
– Filter for companies that are actively running ads (versus historical ads)
– Classify ad type: lead gen vs awareness vs retargeting
– Export active lead-gen advertisers to an enrichment and sequencing workflow
STACK
– SerperDev API + Google News RSS — real-time news monitoring at keyword level
– Clay — data parsing, enrichment, ICP scoring, and sequence routing
– Meta Ads Library, Google Ads Transparency Center, LinkedIn Ads Library — public ad signal
sources
– Indeed, LinkedIn Jobs — job posting signals for intent detection
3.3 Inbound-Led Outbound

Inbound-led outbound is the practice of identifying companies that are already engaging with your
owned assets — primarily your website — and treating that engagement as a high-intent signal to
trigger outbound outreach. A visitor who spent four minutes on your pricing page is not equivalent
to a cold prospect. They have already self-selected into a category of awareness that most of your
outbound list has not reached. The question is whether you can identify who they are and reach them
before the window closes.
Tools like RB2B and Clearbit Reveal de-anonymize website traffic by matching IP addresses to
company records, providing company name, industry, size, and sometimes individual visitor
identity. This data is piped into a Clay workflow where it is enriched, ICP-scored, and routed to an
outbound sequence if the company qualifies. The sequence is materially different from cold
outbound — it references the visit as context, and the message is warmer in tone because the
prospect has already demonstrated awareness.
The economics of inbound-led outbound are significantly better than cold outbound for companies
with meaningful website traffic. Reply rates are higher because relevance is higher. The effort per
meeting is lower because the prospect has self-qualified. The conversion from outreach to meeting
is typically two to three times the rate of cold outbound on the same ICP. This makes website traffic
one of the highest-ROI signal sources available to a GTM engineer, particularly for companies with
inbound content strategies.
WORKFLOW ARCHITECTURE
– Install RB2B or Clearbit Reveal on the company website for IP-to-company de-anonymization
– Route de-anonymized visitor data to Clay via webhook on each page visit event
– Enrich company record: decision-maker identification, ICP scoring, intent classification
– If ICP-qualified: route to warm outbound sequence with visit context in the opening line
– If not ICP-qualified: route to a lower-priority nurture list for periodic re-engagement
– Suppression logic: exclude existing customers, open opportunities, and recently contacted
companies
STACK
– RB2B or Clearbit Reveal — website visitor de-anonymization
– Clay — enrichment, ICP scoring, sequence routing
– Smartlead or Apollo — sequence execution
– HubSpot or Salesforce — suppression list management (existing customers, open opps)

SECTION 4 — ENTERPRISE GTM LAYER
The enterprise GTM layer is where a GTM engineer transitions from delivering outbound
campaigns to delivering revenue systems. The difference is fundamental. A campaign is a one-time
initiative: here is the list, here is the copy, here is the sequence. A revenue system is infrastructure: it
runs continuously, improves over time as new data enters it, and generates pipeline without
requiring repeated human setup. Enterprise companies pay for systems, not campaigns. The
playbooks in this section are the service offerings that command $10K per month retainers rather
than per-meeting fees.
4.1 PLG Playbook
Product-led growth companies have an asset that most outbound programs ignore: a database of
users who have already experienced the product. Free tier users, trial accounts, and freemium
sign-ups represent a pipeline of prospects who have crossed the activation threshold — they
understand what the product does, they have invested time in setting it up, and they have not yet
converted to paid. This is the highest-intent prospect list a B2B company can have, and most
companies manage it with manual sales processes or no process at all.
Clay is one of the most effective PLG enrichment tools available. The workflow begins by
connecting the product database or CRM to a Clay table — this can be done via API, CSV export on
a schedule, or a native integration depending on the CRM. Each sign-up record is enriched with
firmographic data (company size, industry, technology stack, funding stage), behavioral signals
from the product if available (feature usage, session frequency, team expansion), and external
signals (recent news, job postings, ads activity) that indicate whether the company is in a growth
phase. This combined data feeds an upgrade intent score.
The scoring logic for PLG is domain-specific, but the general principle is: accounts that show deep
feature engagement combined with company growth signals are more likely to convert than
accounts that signed up and went dormant. An account where three team members have joined the
free tier in the past 30 days, the company has posted two new engineering roles, and the account has
engaged with pricing-adjacent features is a high-priority conversation. An account that signed up six
months ago, has one user, and has not logged in recently is a re-engagement candidate, not a sales
candidate.
PLG ENRICHMENT AND SCORING ARCHITECTURE
– Connect product DB or CRM sign-up records to Clay via API or scheduled export

– Enrich each account: company size, industry, funding stage, technology stack, headcount growth
– Layer in product behavioral signals if available: feature usage depth, session frequency, team
expansion, pricing page views
– Add external signals: recent news, job postings, active advertising
– Build upgrade intent score combining enrichment signals and behavioral data
– Route high-intent accounts to an AE-assisted sequence (not automated — they are warm)
– Route medium-intent accounts to a nurture sequence with product education content
– Route dormant accounts to a re-activation sequence highlighting new features
SERVICE PACKAGING
For agencies and consultants, the PLG enrichment workflow is a high-value service because it
directly connects to revenue. The deliverable is a scored, enriched sign-up database with automated
routing logic — a system that surfaces upgrade candidates to the sales team in real time without
manual intervention. Pricing for this as a standalone project is typically in the $5K–$15K range for
setup, with an ongoing management fee for data refresh and scoring model maintenance.
4.2 CRM Playbook
Most CRMs are liabilities masquerading as assets. Contact records that were accurate 18 months
ago are now stale — people have changed jobs, companies have been acquired, email addresses
have bounced, phone numbers have changed. The average B2B CRM loses 25 to 30% of its data
quality per year through natural attrition. A CRM with 10,000 contacts that has not been maintained
contains roughly 2,500 to 3,000 contacts whose primary information is wrong.
The CRM playbook uses Clay to transform a stale, incomplete database into an actively maintained
sales asset. The workflow connects the CRM bidirectionally to Clay — records export from
HubSpot or Salesforce to Clay for enrichment, and enriched data writes back to the CRM on a
defined schedule. Each record passes through a re-enrichment waterfall: Lead Magic for email
verification and revalidation, Claygent for job title and company confirmation, and standard
enrichment providers for firmographic data refresh. Records that fail validation are flagged for
human review rather than automatically deleted.
The more powerful application of the CRM playbook is dormant contact re-activation. A CRM full
of contacts that did not convert 18 months ago contains prospects who may now be in-market, may
have changed companies, or may have changed titles to something more relevant. A re-activation

campaign built on Clay identifies three categories of dormant contacts worth re-engaging: (1)
contacts whose company has shown new signals (funding, executive hire, technology adoption)
since the last contact, (2) contacts who have personally changed jobs since the last contact and now
hold a more relevant role at a different company, and (3) contacts whose company has been
acquired by a company already in the pipeline as a prospect.
RE-ENRICHMENT WATERFALL
– Export CRM contacts to Clay via API or scheduled CSV
– Email revalidation via Lead Magic — flag contacts with invalid or catch-all emails
– Job title and company verification via Claygent — identify contacts who have moved or been
promoted
– Firmographic refresh via enrichment providers — update company size, funding stage, technology
stack
– Signal detection pass — check for new events (funding, hiring, news) since last contact
– Write enriched data back to CRM with updated fields and a re-enrichment date stamp
DORMANT RE-ACTIVATION SCORING
– New signal at known company (funding event, executive hire, tech adoption) — high priority
– Contact changed jobs to a relevant role at a different target company — high priority
– Contact's company acquired by or merged with a current pipeline account — high priority
– Contact inactive for 6–12 months with no new signals — low priority, nurture only
SERVICE PACKAGING
The CRM cleanup project is one of the most common enterprise GTM engagements. The
deliverable is a refreshed, validated CRM with documented enrichment logic, plus a Clay workflow
that maintains data quality on an ongoing basis. A CRM audit and cleanup project typically prices at
$3K–$8K for initial setup, with a monthly maintenance fee of $1K–$3K depending on database size
and refresh frequency.
4.3 Job Change Playbook
The moment a new executive joins a target company is one of the most reliable buying signals in
B2B sales. New executives — particularly in roles like VP of Sales, VP of Marketing, CRO, CMO,
and Head of Revenue Operations — enter with a mandate to improve results, a 90-day window to

demonstrate impact, and a budget to act. They are not anchored to the existing vendor relationships
of their predecessor. They have not yet made commitments to competing tools. And they are
actively looking for resources that will help them succeed quickly.
The job change playbook builds a monitoring system that detects these appointments in real time
and routes them to a targeted outreach sequence before competitors have time to act. LinkedIn's job
change signals are the primary data source — when a relevant executive changes roles, LinkedIn
notifies their network within days. Clay's signal monitoring can be configured to detect these
changes across a defined list of target accounts and pipe them automatically into an enrichment and
sequencing workflow.
The outreach angle for new executive contacts is fundamentally different from standard cold
outbound. The message does not open by pitching a product. It opens by acknowledging the
transition, demonstrating awareness of the challenges the new role brings, and offering something of
genuine value — a benchmark report, a competitive analysis, a framework for the 90-day plan. The
goal is to be positioned as a resource before the executive has made any vendor decisions, not to win
a sales conversation that has not yet started.
JOB CHANGE SIGNAL ARCHITECTURE
– Define the list of target accounts and relevant job titles to monitor
– Configure Clay signal monitoring for LinkedIn job changes across the target account list
– Enrich new contacts: verify current role, find email and LinkedIn URL, confirm ICP fit
– Route to a new-executive sequence with a context-aware opening ("Saw you recently joined X as
VP of Sales")
– Suppress contacts who are already in the pipeline or are existing customers
COMBINING WITH FUNDING SIGNALS
The highest-intent combination is a job change signal paired with a funding signal at the same
company. A company that raised a Series B last month and just hired a new VP of Sales is in
maximum growth mode. Both the budget and the mandate to spend it are present. Clay workflows
can detect both signals simultaneously and flag accounts that match both criteria as highest-priority
outreach.
4.4 Social Signals Playbook

Social media engagement is an underused signal layer in B2B GTM. When a decision-maker posts
about a problem, engages with content about a solution category, or comments on a competitor's
post, they are demonstrating awareness and interest in a way that is publicly observable. Unlike
intent data purchased from third-party providers, social signals are free, real-time, and directly
attributable to specific individuals rather than anonymous company-level patterns.
The most effective social signal approach for LinkedIn is post reactor scraping. When a relevant
post — about outbound automation, revenue operations, cold email strategy, or any topic adjacent to
your product — receives engagement, the list of people who liked, commented, or reposted
represents a self-selected audience of people who have demonstrated explicit interest. Clay
workflows that scrape LinkedIn post reactors (using Claygent or a dedicated LinkedIn scraper),
filter by ICP criteria, and route qualified contacts to a personalized outreach sequence can generate
highly engaged prospect lists with minimal acquisition cost.
The outreach angle when using a social signal as context is more natural than a cold pitch. Instead of
opening with a generic value proposition, the message references the specific signal: "Saw you
engaged with X's post about outbound automation last week — we've been building in this space
and thought this might be relevant." This type of opening demonstrates that the outreach is not
automated (even when it partially is), that the sender is paying attention, and that the message is
contextually relevant. These three qualities drive reply rates significantly above cold baseline.
SOCIAL SIGNAL SOURCES
– LinkedIn post reactors — people who engaged with relevant posts in your category
– LinkedIn comment threads on competitor content — people actively discussing your solution
space
– LinkedIn group activity — members who post or comment in relevant industry groups
– Twitter/X keyword engagement — relevant for consumer-adjacent B2B and tech markets
SIGNAL-TO-SEQUENCE WORKFLOW
– Identify high-engagement posts relevant to your product category
– Scrape reactor lists using Claygent or a LinkedIn scraper
– Filter by ICP criteria: job title, company size, industry
– Enrich with email and additional firmographic data
– Build outreach with the specific post as context in the opening line

– Route to LinkedIn connection + email sequence (the LinkedIn connection is warm because you
share a mutual area of interest)
4.5 Conference & Event Playbook
Conferences and events create the most powerful context window in B2B outreach. Everyone
attending a conference is in the same place, sharing the same reference points, and open to
professional connection in a way that cold email recipients are not. A message that references a
shared event is not cold — it is contextual. The challenge is that most companies treat events as
passive networking opportunities rather than engineered pipeline moments. The GTM engineering
approach turns every event — attended or not — into an automated prospecting campaign.
The pre-event play is the highest-leverage phase because it establishes contact before the event
creates noise. The workflow requires an attendee list, which can be sourced from LinkedIn event
pages (for LinkedIn-promoted events), the event's own app (often exports attendee lists to exhibitors
or sponsors), or directly from the event organizer if you are a sponsor. Once the list is enriched and
ICP-filtered, pre-event outreach goes out 10 to 14 days before the event referencing the conference
as context and proposing a meeting at a specific time and place. Pre-event calendar time fills faster
than post-event follow-up because schedules are open and the event creates a natural reason to meet.
The during-event play focuses on real-time engagement and meeting facilitation. For teams
attending the event, Clay workflows push new enriched contacts to a mobile-accessible inbox so
SDRs can follow up immediately after meeting someone. QR codes on event materials capture
contact information and feed directly into a Clay enrichment workflow, triggering an immediate
personalized follow-up while the interaction is still fresh.
The post-event play is the most commonly executed but least differentiated. Every vendor at the
conference sends a follow-up email in the week after the event. The GTM engineering version of
post-event follow-up segments by engagement level (met in person vs saw the booth vs downloaded
a resource vs none) and sends different sequences to each segment. Contacts who had a substantive
conversation receive a personalized follow-up referencing the specific discussion. Contacts who
only visited the booth receive a softer touch with value content. Contacts who appeared on the
attendee list but were not engaged receive a cold-equivalent outreach using the event as context.
EVENT PLAYBOOK PHASES
– Pre-event (10–14 days before): source attendee list, enrich and filter by ICP, send meeting request
with event context, book calendar time before the event fills

– During-event: real-time contact enrichment and immediate follow-up for in-person interactions,
QR code capture feeding Clay workflows
– Post-event (within 48 hours): segment by engagement level, send differentiated sequences by
segment, personalize based on specific interaction where applicable
4.6 Ex-Champions Playbook
Former customers and internal champions who have moved to new companies represent the
highest-converting outbound targets available. They have already done the work of learning your
product, understanding its value, and overcoming the internal objections required to adopt it. When
they move to a new company, they bring that context with them — and they bring a mandate to
solve the same problems they solved at their previous employer.
The mechanics of the ex-champion playbook are straightforward but require consistent CRM
hygiene as a prerequisite. The program begins by identifying every contact in the CRM who was
marked as a champion, key contact, or strong advocate at any point in the relationship. This list is
the foundation. A Clay workflow then monitors this list for job change signals on a regular basis —
weekly is sufficient for most companies. When a champion changes to a new role, the workflow
evaluates whether the new company matches the ICP, enriches the new contact record, and routes
the match to a warm re-engagement sequence.
The warm re-engagement sequence for ex-champions is not a sales pitch. It acknowledges the
relationship, the history, and the transition. The message tone is collegial rather than commercial:
"Saw you joined X as [new title] — congrats on the move. We worked together at [previous
company] and I wanted to reconnect and see how things are going." From this opening, the
conversation naturally progresses to whether the new company has the same challenges that the
product addressed at the previous company. The close rate on ex-champion conversations is
significantly higher than cold outbound because the trust and context are already established.
CHAMPION TRACKING ARCHITECTURE
– Identify all CRM contacts tagged as champions, key contacts, or strong advocates
– Export champion list to Clay for continuous job change monitoring
– When a champion changes to a new role: verify new company ICP fit, enrich new contact record,
flag for warm outreach
– Trigger warm re-engagement sequence with relationship context in the opening

– Suppress contacts currently employed at existing customers or open opportunities
AUTOMATION OF THE TRIGGER FLOW
The full trigger-to-sequence flow can be automated: Clay detects the job change via LinkedIn
signal, checks ICP fit against the scoring model, suppresses against the exclusion list, and enrolls the
contact in the appropriate sequence — all without human intervention. The only human touch is
reviewing the sequence to confirm tone is appropriate before sends begin, and managing the
conversation once replies come in.
4.7 Outbound at Scale — Full TAM Playbook
The full TAM playbook is the enterprise version of the mapping process described in Section 1.
Where the 80/20 TAM process builds a single list for a single campaign, the full TAM playbook
builds a living data infrastructure that covers every account and every relevant decision-maker in a
defined market, keeps that data current automatically, and routes prospects to the appropriate
outreach motion based on their tier and current signal state.
The foundational insight of the full TAM playbook is that mapping accounts and decision-makers is
the highest-ROI GTM investment a company can make, because every subsequent outbound action
— every email sent, every call made, every LinkedIn message sent — is more precise and more
effective when it starts from a complete, accurate, current picture of the market. A company that
knows every company in its TAM, what technology each one uses, who the relevant
decision-makers are, and what signals each account has shown in the past 90 days does not need to
guess who to call. The work of prospecting is replaced by the work of execution.
At enterprise scale, the TAM is segmented into three tiers based on strategic value and propensity to
buy. Tier 1 accounts — typically the top 50 to 100 accounts in the TAM — receive white-glove
treatment: every decision-maker is mapped by persona, the account is enriched on a weekly basis,
and a custom ABM campaign is built for each account individually. Tier 2 accounts receive
automated treatment with personalization at the segment level rather than the account level. Tier 3
accounts receive periodic automated outreach triggered by signals rather than scheduled cadence.
Multi-persona mapping is the capability that differentiates a full TAM playbook from a standard
campaign. For any given account, the champion (the person who will advocate for the product
internally), the economic buyer (the person who controls the budget), and the end user (the person
who will use the product day to day) are different people with different priorities and different
objections. A GTM engineer building a full TAM playbook maps all three personas for Tier 1

accounts and builds outreach sequences specific to each persona's role in the buying process.
TAM AS LIVING INFRASTRUCTURE
– Full TAM list refreshed weekly using Clay's automated enrichment workflows
– Signal monitoring across all accounts for news, job changes, technology adoption, and funding
events
– Account tiering updated dynamically based on signals received — an account that raises a funding
round moves from Tier 3 to Tier 1
– Decision-maker mapping maintained with job change monitoring ensuring accuracy
ACCOUNT TIER TREATMENT
– Tier 1 (top 50–100 accounts): individual ABM campaigns, every decision-maker mapped by
persona, weekly enrichment refresh, executive-level outreach
– Tier 2 (accounts 101–1000): segment-level personalization, automated sequences, monthly
enrichment refresh, signal-triggered escalation to Tier 1
– Tier 3 (remaining TAM): signal-triggered outreach only, no scheduled cadence, quarterly
enrichment refresh
PRICING AS A SERVICE
The full TAM playbook as an agency offering typically prices as a recurring retainer at $5K–$15K
per month, depending on TAM size, number of tiers, and frequency of enrichment refresh. The
value proposition to enterprise clients is that the system replaces the equivalent of a 2–3 person
RevOps team's manual prospecting work, while producing higher data quality and more consistent
execution.
4.8 GTM Analysis & Premium Positioning
The GTM Analysis Clay flow is the diagnostic tool that enables premium positioning. Rather than
arriving at a client conversation with a generic pitch about outbound automation, a GTM engineer
who has run a company through this analysis arrives with a specific assessment: here is what your
current GTM motion looks like, here are the highest-leverage inefficiencies, here are the three
playbooks that would have the most impact on your pipeline in the next 90 days. This is the
difference between selling a service and selling a system.

The analysis uses a company URL as the only required input. From that starting point, a Clay
workflow enriches the company record across 10 or more signal dimensions: CRM type (does the
job description mention Salesforce, HubSpot, or a lesser-known tool, which indicates the technical
sophistication of the RevOps function), website traffic and growth trend (via SimilarWeb or
BuiltWith integrations), PLG vs SLG motion (does the company have a free tier, a trial, or is it
sales-led only), number of salespeople (LinkedIn headcount filter by role), event presence (are they
sponsoring or attending industry conferences), tech stack (particularly for outbound tools — are
they already using Clay, Smartlead, Apollo, or equivalent), active advertising (what and where are
they advertising), recent news events, and funding stage. The output is a scored assessment of which
of the eight enterprise playbooks from this section would have the highest ROI for this specific
company.
The positioning shift this enables is substantial. A GTM engineer who charges $500 for a Clay
outbound campaign is competing on price in a crowded market. A GTM engineer who charges
$10,000 per month for a GTM systems retainer — covering CRM enrichment, TAM management,
signal monitoring, and one active playbook per quarter — is competing on value in a much less
crowded market. The same skills produce the same work; the packaging determines the pricing.
The transition from campaign vendor to GTM advisor requires three things: a diagnostic
methodology (the GTM analysis flow provides this), a portfolio of playbooks that can be deployed
against the findings (the eight playbooks in this section), and the ability to scope, price, and manage
a project that is measured in revenue impact rather than meetings booked. A GTM engineer who can
say "I analyzed your current GTM motion and identified three specific inefficiencies that are costing
you approximately X meetings per month — here is a 90-day plan to address them, with a total cost
of Y and a projected ROI of Z" is not competing with cold email agencies. They are competing with
RevOps consultants and systems integrators.
GTM ANALYSIS SIGNAL DIMENSIONS
– CRM type and apparent technical sophistication — inferred from job postings and integrations
– Website traffic volume and trend — indicates inbound motion strength and product-market fit
stage
– PLG vs SLG motion — free tier, trial, or sales-led only determines the appropriate playbook set
– Sales headcount — number of SDRs, AEs, and RevOps personnel indicates scale of outbound
investment
– Event presence — sponsorship and attendance patterns indicate ABM appetite and budget

– Current outbound tool stack — Clay, Smartlead, Apollo, or manual indicates GTM engineering
maturity
– Active advertising — type, platform, and spend level indicates growth posture
– Recent funding or news events — indicates available budget and current growth mandate
– Technology stack via BuiltWith — integration compatibility and technical infrastructure
sophistication
– Competitor usage — whether they are using your competitors indicates category awareness and
evaluation readiness
PRICING ARCHITECTURE FOR PREMIUM POSITIONING
– GTM analysis and roadmap: $2,500–$5,000 one-time diagnostic (often waived when converted to
retainer)
– Single playbook implementation: $5,000–$10,000 setup + $2,000–$3,000/month maintenance
– Full GTM systems retainer (TAM management + 1 active playbook + CRM enrichment):
$8,000–$15,000/month
– Enterprise transformation project (full TAM playbook + multi-persona mapping + 3+ active
playbooks): $25,000–$50,000 project fee + ongoing retainer
END OF DOCUMENT
The 80/20 GTM Engineering Playbook — Synapse Academy — 2026