try:
    import mmf_setup
    mmf_setup.nbinit()
except ImportError:
    import subprocess
    import sys
    from IPython.display import HTML, Javascript, display
    
    display(HTML(r"""
<style>
figure {
    display: inline-block;
    width: 100%;
    max-width: 45em;
}

figure img {
    align: center;
}

figure figcaption {
    text-align: center;
}

.grade {
   background-color: #66FFCC;
}
</style>"""))
    display(Javascript(r"""// MathJaX customization, custom commands etc.
console.log('Updating MathJax configuration');
MathJax.Hub.Config({
  "HTML-CSS": {
      //availableFonts: ["Neo-Euler"], preferredFont: "Neo-Euler",
      //webFont: "Neo-Euler",
      //scale: 85, // Euler is a bit big.
      mtextFontInherit: true,
      matchFontHeight: true,
      scale: 90, // STIX is a bit big.

  },
  // This is not working for some reason.
  "TeX": {
    Macros: {
        d: ["\\mathrm{d}"],
        I: ["\\mathrm{i}"],
        vect: ["\\vec{#1}", 1],
        uvect: ["\\hat{#1}", 1],
        abs: ["\\lvert#1\\rvert", 1],
        Abs: ["\\left\\lvert#1\\right\\rvert", 1],
        norm: ["\\lVert#1\\rVert", 1],
        Norm: ["\\left\\lVert#1\\right\\rVert", 1],
        ket: ["|#1\\rangle", 1],
        bra: ["\\langle#1|", 1],
        Ket: ["\\left|#1\\right\\rangle", 1],
        Bra: ["\\left\\langle#1\\right|", 1],
        braket: ["\\langle#1\\rangle", 1],
        op: ["\\mathbf{#1}", 1],
        mat: ["\\mathbf{#1}", 1],
        pdiff: ["\\frac{\\partial^{#1} #2}{\\partial {#3}^{#1}}", 3, ""],
        diff: ["\\frac{\\d^{#1} #2}{\\d {#3}^{#1}}", 3, ""],
        ddiff: ["\\frac{\\delta^{#1} #2}{\\delta {#3}^{#1}}", 3, ""],
        Tr: "\\mathop{\\mathrm{Tr}}\\nolimits",
        erf: "\\mathop{\\mathrm{erf}}\\nolimits",
        erfi: "\\mathop{\\mathrm{erfi}}\\nolimits",
        sech: "\\mathop{\\mathrm{sech}}\\nolimits",
        order: "\\mathop{\\mathcal{O}}\\nolimits",
        diag: "\\mathop{\\mathrm{diag}}\\nolimits",
        floor: ["\\left\\lfloor#1\\right\\rfloor", 1],
        ceil: ["\\left\\lceil#1\\right\\rceil", 1],

        mylabel: ["\\label{#1}\\tag{#1}", 1],
        degree: ["^{\\circ}"],
    },
  }
});

// Jupyter.notebook.config.update({"load_extensions":{"calico-document-tools":true}});
// Jupyter.notebook.config.update({"load_extensions":{"calico-cell-tools":true}});
// Jupyter.notebook.config.update({"load_extensions":{"calico-spell-check":true}});
"""))        
    display(HTML(r"""<script id="MathJax-Element-48" type="math/tex">
\newcommand{\vect}[1]{\mathbf{#1}}
\newcommand{\uvect}[1]{\hat{#1}}
\newcommand{\abs}[1]{\lvert#1\rvert}
\newcommand{\norm}[1]{\lVert#1\rVert}
\newcommand{\I}{\mathrm{i}}
\newcommand{\ket}[1]{\left|#1\right\rangle}
\newcommand{\bra}[1]{\left\langle#1\right|}
\newcommand{\braket}[1]{\langle#1\rangle}
\newcommand{\op}[1]{\mathbf{#1}}
\newcommand{\mat}[1]{\mathbf{#1}}
\newcommand{\d}{\mathrm{d}}
\newcommand{\pdiff}[3][]{\frac{\partial^{#1} #2}{\partial {#3}^{#1}}}
\newcommand{\diff}[3][]{\frac{\d^{#1} #2}{\d {#3}^{#1}}}
\newcommand{\ddiff}[3][]{\frac{\delta^{#1} #2}{\delta {#3}^{#1}}}
\DeclareMathOperator{\erf}{erf}
\DeclareMathOperator{\Tr}{Tr}
\DeclareMathOperator{\order}{O}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\sgn}{sgn}
</script>
<i>
<p>This cell contains some definitions for equations and some CSS for styling the notebook.  If things look a bit strange, please try the following:

<ul>
  <li>Choose "Trust Notebook" from the "File" menu.</li>
  <li>Re-execute this cell.</li>
  <li>Reload the notebook.</li>
</ul>
</p>
</i>
"""))

    try:
        HGROOT = subprocess.check_output(['hg', 'root']).strip()
        if HGROOT not in sys.path:
            sys.path.insert(0, HGROOT)
    except subprocess.CalledProcessError:
        # Could not run hg or not in a repo.
        pass
