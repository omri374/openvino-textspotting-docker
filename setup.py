from setuptools import setup, find_packages


setup(
    name="openvino-text-spotting",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
    entry_points={
        'console_scripts':
            [
                'text_spotting=text_spotting.server:main',
                'text_spotting_get_models=text_spotting.model_handler:download_models',
                'get_ocr=call_service:call_ocr'
            ],
    },
)
