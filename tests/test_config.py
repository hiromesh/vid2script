from vid2script.config import Settings, import_services


def test_import_services_does_not_fail_without_matching_keys(tmp_path):
    source = tmp_path / "source.env"
    source.write_text("OTHER=1\n")
    target = tmp_path / "target.env"
    assert import_services(source, target) == []
    assert not target.exists()


def test_import_services_copies_only_missing_service_keys(tmp_path):
    source = tmp_path / "source.env"
    source.write_text('OSS__BUCKET="new"\nVIDEO_ANALYSIS__ASR_API_KEY="key"\n'
                      'ASR_SEGMENT_SECONDS="600"\nOTHER="ignored"\n')
    target = tmp_path / "target.env"
    target.write_text('OSS__BUCKET="old"\n')
    assert import_services(source, target) == ["VIDEO_ANALYSIS__ASR_API_KEY", "ASR_SEGMENT_SECONDS"]
    text = target.read_text()
    assert 'OSS__BUCKET="old"' in text and "OTHER" not in text


def test_settings_load_reads_asr_tuning(tmp_path, monkeypatch):
    for key in ("MODEL", "BASE_URL", "API_KEY", "ASR_SEGMENT_SECONDS", "ASR_POLL_INTERVAL_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("MODEL=m\nBASE_URL=https://example.invalid/v1\nAPI_KEY=k\n"
                   "ASR_SEGMENT_SECONDS=120\nASR_POLL_INTERVAL_SECONDS=3\n")
    settings = Settings.load(env)
    assert settings.asr_segment_seconds == 120
    assert settings.asr_poll_interval_seconds == 3
