"""Operations Context 의 Use Case.

    DeployModel / ReleaseVersion        (5-1, 5-2)
    IngestInferenceLog                  (5-3)
    ObserveHealth                       (5-4 ~ 5-7)
    FindOnset                           (5-4)
    QuarantineDeployment / Resume       (5-8)
    CompareShadow                       (5-9)
    RollbackDeployment                  (5-10)
    DecideRetraining                    (5-11)

이 Layer 는 판단하지 않는다. 측정기를 부르고, Domain Policy 에 건네고, DTO 로 바꾼다.
"""
