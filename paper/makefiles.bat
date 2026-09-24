@echo off
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 exit /b %errorlevel%
