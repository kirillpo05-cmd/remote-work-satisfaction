# Working with an AI assistant

The brief asked where judgement was exercised by the author and where the tool
was simply generating. This document answers that. It is written as the work
happens, not reconstructed afterwards.

## How the tooling was configured

This repository ships a `CLAUDE.md`, three specialised subagents, two
glob-scoped rule sets and one project-specific skill. That is a deliberate
investment: an assistant without project context produces locally correct but
architecturally disconnected code. The configuration is part of the submitted
work.

Two configuration decisions are worth naming. The `reviewer` subagent is
configured without `Write` or `Edit` permissions, but it keeps `Bash` so it can
run the tests, and `Bash` can write files. This is therefore a default
constraint rather than an enforced guarantee. It reduces the chance that the
reviewer starts fixing problems instead of reporting them, since a fixed problem
disappears from the report together with the chance to discuss it. Whether the
reviewer actually stayed read-only was checked by reading the diff after each
review pass, not assumed from the toolset.

The second decision was to run only the statistical and modelling agent on the
larger model. The service and interface layers are mechanical enough for the
smaller model to handle, and the cost difference is noticeable over a long
session.

### Episode 0 — Pandas changed the data before the code even started

The first issue appeared before the actual processing started, during the file loading stage. The specification assumed that missing values in two columns represented the absence of a state or information. However, the raw data did not contain missing values at all. It contained the word `"None"` instead. There were 1196 such values in one column and 1629 in another.

When pandas loaded the file, it automatically treated `"None"` as a missing value. This meant that the data was changed before any of the planned processing steps were performed. Later, the imputer could have changed about one third of the `Physical_Activity` values from "does not exercise" to "Weekly". This would not cause an error, and all tests could still pass. The result would therefore be silent data corruption.

The fix was to use `keep_default_na=False` when loading the file.

The main lesson from this episode is that correct implementation is not enough when the assumptions about the input data are wrong. The assistant could follow the specification exactly and still produce an incorrect result. In this case, the problem was found by inspecting the raw file before trusting the processing pipeline. This leads to the next issue, where the implementation was also correct according to the specification, but the statistical method itself was incomplete.

### Episode 1 — The p-value could become zero

The assistant calculated the p-value as `k/N` because this was exactly what the specification required. However, the observed value itself is one of the possible outcomes under the null hypothesis. Therefore, a permutation test should not return a p-value of exactly zero, because this would imply that the observed result is impossible.

The fix was to use `(1+k)/(1+N)` instead.

The lesson here is similar to the previous episode. The assistant implemented the defined method correctly, but the method described in the specification did not fully account for the statistical situation. A human review was needed to identify this issue and adjust the formula.

This also shows why methodological decisions need to be reviewed separately from implementation. The code can be technically correct while the underlying statistical rule still needs improvement.

### Episode 2 — Which standard deviation should be used?

The next issue came from an unclear rule in the specification. The instruction said that the baseline should be beaten by two standard deviations, but it did not specify which standard deviation should be used.

The assistant used the standard deviation of the model itself and reported the ambiguity. However, because the models were evaluated on the same folds, the more appropriate approach was to use the standard deviation of the paired differences between the models across the folds.

The fix was therefore to calculate the standard deviation of these paired differences.

The main lesson is that when a rule is not fully defined, the assistant can select a defensible interpretation, but the choice should be made visible. In this case, the ambiguity was correctly identified, although the initial default interpretation was not the statistically strongest one.

This episode also shows the value of reviewing assumptions before they become part of the final decision. The same principle becomes even more important in the next episode, where following the specification literally would have produced a meaningless diagnostic.

### Episode 3 — The test itself was not useful

The specification required the permutation test to be performed on the selected model. However, the selected model was a dummy model for which the permutation test was not meaningful, because changing the order of the data would not change its predictions.

Instead of following the instruction literally, the assistant ran the test on the boosting model and asked for confirmation of this deviation.

This was a case where following the purpose of the analysis was more useful than following the exact wording of the specification. At the same time, the assistant did not silently change the method. It identified the conflict and brought it forward for review.

The main lesson is that an assistant can sometimes recognise that a requested procedure does not make sense in the context where it is being used. However, such a deviation should be made explicit and reviewed rather than introduced without notice.

The next episode moved from methodology to testing. Here, the problem was not that the assistant followed the specification incorrectly, but that the successful test results themselves were accepted too quickly.

### Episode 4 — Passing all tests can also be a warning

The assistant initially reported success because all 13 tests passed on the first attempt. However, in a TDD process, tests are normally written before the implementation, so they should initially fail. If all tests pass immediately, this can be a warning sign that the tests are not actually checking the intended behaviour.

To verify this, I asked the assistant to identify a specific code mutation that would make each test fail. I also asked it to apply two of these mutations in practice.

All 13 tests had a possible mutation that caused them to fail. This showed that the tests were capable of detecting changes. However, the process also revealed a real problem: the split test was reading the real CSV file.

The main lesson is that a green test suite only proves that the current implementation satisfies the current tests. It does not prove that the tests themselves are strong enough. Testing the tests through mutations was therefore necessary to understand whether the validation process was actually reliable.

This concern about validation also connects to the next issue. Even a technically correct implementation and a working test suite cannot guarantee that the chosen method is able to detect the type of relationship that the analysis is looking for.

### Episode 5 — The screening method could miss an important relationship

The assistant encoded a feature using only one column because this was required by the specification. However, this representation mainly allows the method to detect monotonic relationships.

For example, a U-shaped relationship may have poor results at both ends and good results in the middle. A single-column representation could treat this as almost no relationship and classify the feature as noise. For a product whose main purpose is to distinguish signal from noise, this could lead to a serious false negative.

The fix was to use one-hot encoding for the different levels and to add tests for U-shaped relationships in both directions.

The main lesson is that a method can be implemented exactly as designed and still be unable to detect important patterns in the data. This type of problem cannot usually be found by tests or linters, because the code is behaving as specified. It requires reasoning about what the statistical method is actually capable of seeing.

The next gate showed why this distinction between model behaviour and real evidence was important for the final decision.

### Episode 6 — Feature importance is not the same as a final decision

In Gate 3, `Years_of_Experience` was the most important feature in both models, and its interval did not include zero. However, the screening method classified it as noise, and both models performed worse than the dummy baseline.

The final decision was to reject the feature as a reliable signal. There were several reasons for this. First, the models were not independent sources of evidence because they used the same data and could have learned from the same coincidence. Second, the interval measured only the variation caused by the permutation process. Third, the models themselves did not agree strongly, with a correlation of approximately 0.19.

In addition, 32% of the rows were identified as impossible or invalid, which further reduced the meaning of the results.

The main lesson is that model feature importance is not, by itself, evidence of a real relationship or influence. The decision was based on the complete set of diagnostics rather than on a single importance score. In this case, the rule defined before the implementation was supported by the actual results.

This episode also demonstrated that decisions should be based on evidence from the whole pipeline. The final episode showed the same principle at the deployment level, where a standard engineering practice was changed after measuring its actual effect.

### Episode 7 — The deployment decision changed after measuring the real performance

The original decision was not to store generated artifacts in git, following the standard practice of keeping generated files out of version control.

However, a clean `docker compose up` took approximately 62 minutes because the system had to calculate everything again from the beginning. This showed that the standard rule was not suitable for the actual deployment requirements.

The solution was to commit the artifacts and include them in the Docker image. The entrypoint was also changed so that the system recalculates the artifacts only when they are not already available.

The main lesson is that standard engineering practices should be evaluated against the actual requirements of the system. In this case, the artifacts had already been shown to be reproducible, so storing them was considered safe. The measured deployment time provided a clear reason to change the original decision.

Overall, this episode reinforced a broader principle from the project: decisions should be based on evidence and the actual purpose of the system rather than on general rules alone.

### How the tools were used

The work was divided between implementation and methodological decisions. The assistant wrote most of the actual code for the different system layers. I was responsible for writing the specification, making the methodological decisions, reviewing the gates, checking whether the conclusions were supported by the data, and writing the main text in the three files.

The git history also shows this separation. The specification and methodological decisions were committed before the implementation started.

This separation was important because the main risks in the project were not related to writing code. They were related to whether the assumptions, statistical methods, and evaluation criteria were correct in the first place.

### Where the assistant helped the most

The assistant was most useful for tasks that were repetitive, mechanical, and clearly defined. This included writing boilerplate code, creating tests from the edge-case table, preparing the Dockerfile, defining Pydantic contracts, and implementing the permutation and bootstrap logic.

These tasks involved a significant amount of code but relatively little methodological decision-making. The assistant could therefore complete them quickly and consistently, which reduced the amount of time spent on routine implementation work.

### Where the assistant was less useful or potentially harmful

The most important problems were related to system design and methodology rather than coding.

In three cases, the assistant correctly implemented a specification that was itself wrong or incomplete. These were Episodes 0, 3, and 5. The code followed the specification, so standard tests and linters could not identify the underlying problems.

This is one of the main limitations of using an assistant for this type of work. If the specification contains an incorrect assumption, the assistant can implement that assumption accurately and efficiently without necessarily questioning it.

For this reason, the specification went through five rounds of revision before the first line of code was written. This process was necessary because the assistant did not identify that three incompatible effect scales could not be combined into a single meaningful ranking. Without human methodological review, it could have produced a technically correct implementation of a fundamentally flawed design.

Overall, the assistant was highly effective for implementation, repetitive work, and translating clear decisions into working code. Its main limitation was in challenging the assumptions behind the methodology. The human review was therefore most valuable in situations where the code looked correct but the underlying reasoning, data assumptions, or statistical design were not.
