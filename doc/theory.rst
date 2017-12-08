
Theory
========================

The monotone classification algorithms implemented here are described in the paper paper [bartley2017]_. A series of instance based rules based on inequalities in monotone features, and cone based constraints on non-monotone features, are fitted using gradient boosting. The resulting binary classifier takes the form:

.. math::
    F(\textbf{x})=sign(a_0 + \sum_{m=1}^{M}a_m f_m(\textbf{x}))

where each rule is based on some base point x_m and a conical constraint on the non-monotone features defined by vectors v_m and w_m, of the form:

.. math::
    f_m(\textbf{x},\textbf{z})= \textbf{1}  \big[ \textbf{x}\succeq \textbf{x}_m \: \land \: \textbf{z} \in\{\textbf{z} \mid  \textbf{w}_m^T\Delta\textbf{z} \le \textbf{v}_m^T\Delta\textbf{x}  \} \big]




.. [bartley2017] Bartley C., Liu W., Reynolds M. (2017). 
A Novel Framework for Partially Monotone Rule
Ensembles. ICDE submission, prepub, http://staffhome.ecm.uwa.edu.au/~19514733/

