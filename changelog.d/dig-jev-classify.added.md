The X dig loop now favors broad retrieval and relevance-classifies results:
dig planner feedback tells the planner that bare entity/category terms are
safe because every candidate post is scored by Jev before merging — off-topic
posts are dropped (`jev_rejected` stat, `jev_score` on kept items) instead of
polluting the report. This recovers recalls that keyword-AND queries miss
(e.g. a launch post worded without the search verbs). `LAST30DAYS_X_DIG_JEV=0`
disables the pass; it fails open when no Jev key is configured.
