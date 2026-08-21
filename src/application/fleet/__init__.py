"""Fleet Context 의 Use Case.

    RegisterDevices / IngestUplink      (6-1)
    InspectLakeLayout                   (6-2)
    SummarizeFleet                      (6-3, 6-11)
    BuildTrainingDataset                (6-4)
    SubmitTrainingJob / PollTrainingJob (6-5)
    PublishRelease / PromoteRelease     (6-6, 6-7)
    PlanRollout / AdvanceRollout        (6-8)
    RollbackRollout                     (6-9)
    TraceLineage                        (6-10)

이 Layer 에도 boto3 가 없다. Port 만 본다.
"""
