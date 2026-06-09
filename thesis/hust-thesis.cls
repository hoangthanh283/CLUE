% -------------------------------------------------------------------
%  @LaTeX-class-file{
%     filename        = "hust-thesis.cls",
%     version         = "0.0.1",
%     date            = "2021",
%     codetable       = "ISO/ASCII",
%     keywords        = "LaTeX, hust-thesis",
%     supported       = "send email to ngocjr7@gmail.com",
%     docstring       = "A thesis class for a HUST dissertation. This is an adaptation based on Harvard's template."
% --------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hust-thesis}[2021/05/06 v0.0.1 Hanoi University of Science and Technology Thesis Class]

\newcommand{\mylanguage}{en}

\DeclareOption{en}{\renewcommand{\mylanguage}{en}}
\DeclareOption{vi}{\renewcommand{\mylanguage}{vi}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[12pt, oneside]{book}

% Required packages
\RequirePackage{graphicx}
\RequirePackage{hyperref}
\RequirePackage[toc]{glossaries}
\RequirePackage{glossary-mcols}
\RequirePackage{here}
\RequirePackage[toc]{appendix}

\RequirePackage{amssymb}
\RequirePackage{amsmath}
\RequirePackage{mathtools}
\RequirePackage{mathspec} %loads fontspec as well

\RequirePackage[authoryear, comma, sort&compress]{natbib}
\RequirePackage[small, md, sc]{titlesec}
\RequirePackage[tight,nice]{units}
\RequirePackage{verbatim}
\RequirePackage[palatino]{quotchap}
\RequirePackage{ragged2e}
% colors
\RequirePackage{color}
\definecolor{Crimson}{rgb}{0.6471, 0.1098, 0.1882}

\hypersetup{
	linktocpage,
    colorlinks,
    citecolor=black,
    filecolor=black,
    linkcolor=black,
    urlcolor=black,
}

\RequirePackage{url}
\RequirePackage[titles]{tocloft}

\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{2}

\renewcommand{\cftchapfont}{\normalsize \scshape}
\setlength{\parindent}{12pt}
\setlength{\parskip}{1em}

\RequirePackage{pdfsync} %do pdf synchronization [andy]
\RequirePackage[
    type={CC},
    modifier={by-nc-sa},
    version={3.0},
]{doclicense}

% \usepackage[closeFloats, noSeparatorLine]{fltpage} %use the custom modified fltpage package
% \RequirePackage{afterpage} 

\synctex=1 % turn synctex on automatically to sync between pdf viewer [andy]

%   list an entire bibliography entry inline. Useful for acknowledging when my paper was previously published
\RequirePackage{bibentry} 
\nobibliography*        

% \RequirePackage{lettrine} % big letter at start of chapter
\RequirePackage[a4paper]{geometry}
\geometry{left=3.5cm, right=2.5cm, top=3cm, bottom=3cm}

\RequirePackage{fancyhdr} 
\pagestyle{plain} % options: empty , plain , fancy

% \RequirePackage[palatino]{quotchap} % chaper number 
% \definecolor{chaptergrey}{rgb}{0.6,0,0}

\RequirePackage{afterpage} 

\RequirePackage[parfill]{parskip}
\RequirePackage{titling}
\RequirePackage{setspace} 
\RequirePackage{booktabs} % for much better looking tables
\RequirePackage[figurewithin=section, tablewithin=section]{caption}

% line alignment
\onehalfspacing
\addtolength{\footnotesep}{\footnotesep}
\justifying
%\raggedright

% font
\renewcommand{\rmdefault}{ptm}
\renewcommand{\sfdefault}{phv}


% use Minion Pro instead of standard LaTeX fonts for math display
% \setmathsfont(Latin,Greek)[Numbers={OldStyle,Proportional},
%     ExternalLocation=fonts/,
%     Scale=0.95]{MinionPro-It}
% \setmathsfont(Digits)[Numbers={Proportional}, ExternalLocation=fonts/, Scale=0.95]{MinionPro-Regular}
% \setmathrm[ExternalLocation=fonts/]{MinionPro-Regular}

% use Arno Pro instead of standard LaTeX fonts for math display
\setmainfont{Times New Roman}

% Fonts for titles
\newfontfamily{\arnosubhead}[Path=./fonts/]{Arno Pro Subhead}
\newfontfamily{\arno}[Path=./fonts/]{Arno Pro Smbd}
\newfontfamily{\arnobody}[Path=./fonts/]{Arno Pro}
\newfontfamily{\arnoitalic}[Path=./fonts/]{Arno Pro Italic}

\titleformat{\chapter}[display]
  {\arno\Huge}
  {\chaptertitlename\ \thechapter}{20pt}{\Huge\sloppy}

\titleformat{\section}
  {\arno\Large}
  {\thesection}{1em}{\Large}
  
\titleformat{\subsection}
  {\arno\large}
  {\thesubsection}{1em}{\large}

\titleformat{\subsubsection}[runin]
  {\bf}
  {\thesubsection}{1em}{}

\titleformat{\paragraph}[runin]
  {\it}
  {}{1em}{}


% \setmainfont[ExternalLocation=fonts/]{Arno Pro.otf}%
%   [Ligatures=TeX,
%   BoldFont=Arno Pro Bold.otf,
%   ItalicFont=Arno Pro Italic.otf,
%   BoldItalicFont=Arno Pro BoldItalic.otf]

% \RequirePackage{fontspec,xltxtra,xunicode}
% \defaultfontfeatures{Mapping=tex-text}
% \setromanfont[Mapping=tex-text, 
% 			  Numbers=OldStyle, 
% 			  Ligatures={TeX}, 
% 			  ExternalLocation=fonts/,
% 			  SizeFeatures={%
% 		          {Size={-10}, Font=* Caption},
% 		          {Size={10-14.0}, Font=*},
% 		          {Size={13.01-19.9}, Font=* Subhead, Ligatures=Rare, Contextuals={Swash,Alternate}},
% 		       {Size={19.91-},Font=* Display, Ligatures=Rare, Contextuals={Swash,Alternate}}}]
% 			  {Arno Pro} %

% \setmainfont[ExternalLocation=fonts/]{Arno Pro.otf}%
%   [Ligatures=TeX,
%   BoldFont=Arno Pro Bold.otf,
%   ItalicFont=Arno Pro Italic.otf,
%   BoldItalicFont=Arno Pro BoldItalic.otf]

  
%Set fonts for Mac Specific (Comment out on Windows XP)
%\setsansfont[Scale=MatchLowercase,Mapping=tex-text]{Helvetica Neue} 
%\setmonofont[Scale=MatchLowercase]{Consolas}

% some definitions
\def\degreeyear#1{\gdef\@degreeyear{#1}}
\def\degreemonth#1{\gdef\@degreemonth{#1}}
\def\degree#1{\gdef\@degree{#1}}
\def\advisor#1{\gdef\@advisor{#1}}
\def\department#1{\gdef\@department{#1}}
\def\institute#1{\gdef\@institute{#1}}
\def\field#1{\gdef\@field{#1}}
\def\university#1{\gdef\@university{#1}}
\def\universitycity#1{\gdef\@universitycity{#1}}
\def\universitystate#1{\gdef\@universitystate{#1}}
\def\authoremail#1{\gdef\@authoremail{#1}}
\def\authorphone#1{\gdef\@authorphone{#1}}
\def\specialization#1{\gdef\@specialization{#1}}
\def\major#1{\gdef\@major{#1}}


\newcommand{\isenglish}[3][\mylanguage]{
    \ifthenelse{\equal{#1}{en}}{#2}{#3}
}

\isenglish{
    \RequirePackage[english]{babel}
}{
    \RequirePackage[vietnamese]{babel}
}

\renewcommand{\maketitle}{ 
	\singlespacing
	\thispagestyle{empty}
% 	\vspace*{\fill} 
    % \vspace{150pt} 
	\begin{center}
    \begin{large}\fontsize{15pt}{1}\selectfont \arno\textbf{\MakeUppercase{\@university}} \end{large} \\[5cm]
    {\arno\fontsize{25pt}{1}\selectfont \textbf{\MakeUppercase{\@degree}} \\ [1cm]}
    {\arno\fontsize{23pt}{1}\selectfont \textbf{\@title}}\\[0.635cm]
    {\arno\fontsize{15pt}{1}\selectfont \textbf{\MakeUppercase\@author}}\\[0.1058cm]
    {\fontsize{14pt}{1}\selectfont \textcolor{black}{\href{mailto:\@authoremail}{\@authoremail}}}\\[0.4233cm]
    
    % {\fontsize{15pt}{1}\selectfont \textbf{\isenglish{Field}{Ngành}: \@field}}\\[0.1058cm]
    {\arno\fontsize{15pt}{1}\selectfont \textbf{\isenglish{Major}{Chuyên ngành}: \@major}}\\[3.5cm]
    
    \vbox{
        
    	\begin{table}[H]
    	\begin{tabular}{l c l l} 
    	\fontsize{14pt}{1}\selectfont\textbf{\isenglish{Thesis advisor}{Giảng viên hướng dẫn}}~~&\bf:  & \arnobody\fontsize{14pt}{1}\selectfont\@advisor \hspace{1cm} & \underline{\hspace{4cm}}\\ 
    	& & & {\arnobody\fontsize{11pt}{1}\selectfont \isenglish{Signature of advisor}{Chữ ký GVHD}} \\
    	\arno\fontsize{14pt}{1}\selectfont\textbf{\isenglish{Department}{Bộ môn}}&\bf:  & \multicolumn{2}{l}{\arnobody\fontsize{14pt}{1}\selectfont\@department} \\
    	\arno\fontsize{14pt}{1}\selectfont\textbf{\isenglish{Institute}{Viện}}&\bf: & \multicolumn{2}{l}{\arnobody\fontsize{14pt}{1}\selectfont\@institute}
    	\end{tabular} 
    	\end{table}
    }
    \vspace*{\fill}
	{\arno\fontsize{14pt}{1}\selectfont \textbf{\@universitycity, \@degreemonth-\@degreeyear}}\\
	
	\end{center} 
	\newpage
}

\def\authorclass#1{\gdef\@authorclass{#1}}
\def\duration#1{\gdef\@duration{#1}}
\def\statement#1{\gdef\@statement{#1}}
\def\declaration#1{\gdef\@declaration{#1}}
\def\attestation#1{\gdef\@attestation{#1}}

\attestation{
    \dotline{\linewidth}\newline
    \dotline{\linewidth}\newline
}

\newcommand{\requirementpage}{
    \onehalfspacing
	\thispagestyle{empty}
	\isenglish{
        \begin{center}
        \Large 
        \MakeUppercase{\@degree \space Assignment}
        \end{center}
	}{
        \begin{center}
        \Large 
        \MakeUppercase{Phiếu giao nhiệm vụ \space \@degree}
        \end{center}
    }
    
    \begin{enumerate}
    \item \isenglish{Student's information}{Thông tin sinh viên}:\\
    \isenglish{Name}{Họ và tên}: \@author. \\
    \isenglish{Phone}{Điện thoại}: \@authorphone \quad  Email: \@authoremail\\
    \isenglish{Class}{Lớp}: \@authorclass \\
    \isenglish{Affiliation}{Trường}: \@university.\\
    \isenglish{Duration}{Thời gian thực hiện}: \@duration.
    \item \isenglish{Thesis title}{Tên luận văn}: \@title
    \item \isenglish{Thesis statement}{Mục tiêu của \MakeLowercase{\@degree}}:\\
        \@statement
    \item \isenglish{Declarations/Disclosures}{Lời cam đoan của sinh viên}:\\
        \@declaration
    \\[0.5cm]
    \signbox{\isenglish{Author}{Tác giả}}{\@author}
    
    
    \item Attestation of thesis advisor: \\
    \@attestation
    
    
    \signbox{\isenglish{Thesis Advisor}{Giáo viên hướng dẫn}}{\@advisor}
    \end{enumerate}
    \newpage
}

% You might also consider licensing your work under Creative Commons). See: http://creativecommons.org/weblog/entry/12824 for other PhD students who have released their work under creative commons.

\newcommand{\copyrightpage}{
	\newpage \thispagestyle{empty} \vspace*{\fill}
	{\sc\noindent \copyright~\textit{\@degreeyear \hspace{3pt}~- \theauthor}} \\
	\noindent {\sc All rights reserved.} \\
	\vspace{5pt}
	\noindent Re-distributed by \@university \space under license with the author.
	
	\doclicenseThis
	\vspace*{\fill} \newpage \rm
}

\newcommand{\abstractpage}[2][\mylanguage]{
	\newpage
% 	\pagenumbering{roman}
% 	\setcounter{page}{3}
	\pagestyle{fancy}
	\lhead{\isenglish[#1]{Thesis advisor}{Giảng viên hướng dẫn}: \@advisor} \rhead{\@author}
	\renewcommand{\headrulewidth}{0.0pt} 
	\begin{center}
	\vspace*{1pt}
	\Large \textcolor{Crimson}{\textit{\@title}} \normalsize\\
	\vspace*{15pt}
	\bf \isenglish[#1]{Abstract}{Tóm tắt đồ án} \\ \rm
	\end{center}
	\onehalfspacing %Harvard registrar requests that abstract is double spaced
	#2
	\onehalfspacing %Harvard registrar requests that abstract is double spaced	
	\newpage \lhead{} \rhead{}
	\cfoot{\thepage}
	\onehalfspacing
}

\newcommand{\dedicationpage}[1]{
	\pagestyle{fancy}
	\newpage \thispagestyle{fancy} \vspace*{\fill}
	\sc \noindent #1
	\vspace*{\fill} \newpage \rm
	}

% the list of authors
\newcommand{\authorlist}[1]{
	\pagestyle{fancy}
	\newpage
	\thispagestyle{fancy} 
	\isenglish{\chapter*{Author List}}{\chapter*{Danh sách tác giả}}
	\noindent #1
	\newpage \rm
	}
	

% the acknowledgments page
\newcommand{\acknowledgments}[1]{
    \isenglish{\chapter*{Acknowledgments}}{\chapter*{Lời cảm ơn}}
    \pagestyle{fancy}
	\noindent
	\onehalfspacing
	#1
	\vspace*{\fill} 
	\newpage
}
	
% An environment for paragraph-style section
\providecommand\newthought[1]{%
   \addvspace{1.0\baselineskip plus 0.5ex minus 0.2ex}%
   \noindent\textsc{#1}}

% An environment for sign box
\newcommand{\signbox}[2]{
\begin{minipage}{0.5\linewidth}
\end{minipage}\hfill
\begin{minipage}{0.45\linewidth}
\begin{flushright}
\vbox{
    \begin{center}
	\textit{\@universitycity, \isenglish{date}{ngày} \space \space \space \isenglish{month}{tháng} \space \space \space \isenglish{year}{năm} \@degreeyear}
	\\ #1
	\\~
	\\~
	\\ \textit{#2}
	\end{center}
}
\end{flushright}
\end{minipage}
}

% \dotline and \dotfill
\def\dotfill#1{\cleaders\hbox to #1{.}\hfill}
\newcommand\dotline[2][.5em]{\leavevmode\hbox to #2{\dotfill{#1}\hfil}}

\renewenvironment{savequote}[1][10cm]{%
\begin{lrbox}{\@quotebox}
\noindent
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lr|}
&\begin{minipage}[t]{#1}\footnotesize\quotefont
\ignorespaces
}{%
\end{minipage}
\end{tabular*}
\end{lrbox}
\global\setbox\@quotebox\copy\@quotebox
\global\let\@printcites\@iprintcites
\ignorespacesafterend}


% remove hyperlink, mostly used for citation in caption and headline
\newcommand{\nohyper}[1]{\protect\NoHyper#1\protect\endNoHyper}

\renewcommand\bibname{References}
\renewcommand\listfigurename{Listing of Figures}

% undo the wrong changes made by mathspec
\makeatletter 
\let\RequirePackage\original@RequirePackage
\let\usepackage\RequirePackage
\makeatother

\renewcommand{\glsmcols}{2}
\setglossarystyle{mcolindex}