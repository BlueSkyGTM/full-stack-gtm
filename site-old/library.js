/* ============================================================
   AI SCHOOL · Library
   Curated reading list for AI engineering. English only.
   Filtered from EbookFoundation/free-programming-books.
   Organized by tier: Fundamentals → Intermediate → Advanced.
   All resources free to access.
   ============================================================ */

var LIBRARY = {

  fundamentals: [
    /* ── Mathematics ─────────────────────────────────────── */
    {
      title: "Mathematics for Machine Learning",
      author: "Deisenroth, Faisal, Ong",
      url: "https://mml-book.github.io",
      format: "HTML, PDF",
      topic: "Math",
      note: "Linear algebra, calculus, probability — all through the ML lens. Start here."
    },
    {
      title: "A Programmer's Introduction to Mathematics",
      author: "Jeremy Kun",
      url: "https://pimbook.org",
      format: "HTML",
      topic: "Math",
      note: "Makes abstract math concrete for people who code. Genuinely different from textbooks."
    },
    {
      title: "Linear Algebra Done Right",
      author: "Sheldon Axler",
      url: "https://linear.axler.net",
      format: "HTML",
      topic: "Math",
      note: "The cleanest treatment of linear algebra. Builds intuition, not just mechanics."
    },
    {
      title: "Convex Optimization",
      author: "Boyd, Vandenberghe",
      url: "https://web.stanford.edu/~boyd/cvxbook",
      format: "PDF",
      topic: "Math",
      note: "Stanford. Gradient descent and optimization — the engine behind all of ML."
    },
    {
      title: "Introduction to Probability",
      author: "Grinstead, Snell",
      url: "https://math.dartmouth.edu/~prob/prob/prob.pdf",
      format: "PDF",
      topic: "Math",
      note: "Dartmouth. Clear, complete, free. The probability foundation you actually need."
    },
    {
      title: "Think Stats: Probability and Statistics for Programmers",
      author: "Allen B. Downey",
      url: "https://greenteapress.com/thinkstats/",
      format: "HTML, PDF",
      topic: "Math",
      note: "Teaches stats through Python. No prerequisite math needed."
    },
    {
      title: "Think Bayes: Bayesian Statistics Made Simple",
      author: "Allen B. Downey",
      url: "https://www.greenteapress.com/thinkbayes/",
      format: "HTML, PDF",
      topic: "Math",
      note: "Bayesian thinking through code. Pairs with Phase 01 probability work."
    },
    {
      title: "Bayesian Methods for Hackers",
      author: "Cameron Davidson-Pilon",
      url: "https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/",
      format: "HTML, Jupyter",
      topic: "Math",
      note: "Probabilistic programming in Python. Computational over analytical — the right approach."
    },
    {
      title: "Mathematics for Computer Science",
      author: "Lehman, Leighton, Meyer",
      url: "https://courses.csail.mit.edu/6.042/spring18/mcs.pdf",
      format: "PDF",
      topic: "Math",
      note: "MIT 6.042. Discrete math, graph theory, proofs. The CS foundation."
    },
    /* ── Python ───────────────────────────────────────────── */
    {
      title: "A Whirlwind Tour of Python",
      author: "Jake VanderPlas",
      url: "https://jakevdp.github.io/WhirlwindTourOfPython/",
      format: "HTML, Jupyter",
      topic: "Python",
      note: "The fastest Python ramp-up that exists. Read this before Phase 01."
    },
    {
      title: "Python Data Science Handbook",
      author: "Jake VanderPlas",
      url: "https://jakevdp.github.io/PythonDataScienceHandbook",
      format: "HTML, Jupyter",
      topic: "Python",
      note: "NumPy, Pandas, Matplotlib, Scikit-learn — the scientific Python stack in one book."
    },
    {
      title: "The Hitchhiker's Guide to Python",
      author: "Kenneth Reitz, Tanya Schlusser",
      url: "https://docs.python-guide.org",
      format: "HTML",
      topic: "Python",
      note: "Best practices, tooling, project structure. Opinionated and correct."
    },
    {
      title: "Architecture Patterns with Python",
      author: "Harry Percival, Bob Gregory",
      url: "https://www.cosmicpython.com/book/preface.html",
      format: "HTML",
      topic: "Python",
      note: "Domain-driven design, clean architecture, TDD — applied to Python services."
    },
    {
      title: "Python Programming and Numerical Methods",
      author: "Kong, Siauw, Bayen",
      url: "https://pythonnumericalmethods.berkeley.edu/notebooks/Index.html",
      format: "HTML, Jupyter",
      topic: "Python",
      note: "Berkeley. Scientific Python for engineers. Bridges math and code."
    },
    {
      title: "Research Software Engineering with Python",
      author: "Irving, Hertweck, Johnston, et al.",
      url: "https://merely-useful.tech/py-rse/",
      format: "HTML",
      topic: "Python",
      note: "How to write Python that lasts: testing, packaging, Git, reproducibility."
    },
    /* ── Algorithms ───────────────────────────────────────── */
    {
      title: "Algorithms",
      author: "Jeff Erickson",
      url: "https://jeffe.cs.illinois.edu/teaching/algorithms/book/Algorithms-JeffE.pdf",
      format: "PDF",
      topic: "Algorithms",
      note: "Illinois. The best free algorithms textbook. Rigorous and readable."
    },
    {
      title: "Algorithms, 4th Edition",
      author: "Sedgewick, Wayne",
      url: "https://algs4.cs.princeton.edu/home/",
      format: "HTML",
      topic: "Algorithms",
      note: "Princeton. Data structures and algorithms implemented clearly."
    },
    {
      title: "Foundations of Computer Science",
      author: "Aho, Ullman",
      url: "http://infolab.stanford.edu/~ullman/focs.html",
      format: "HTML",
      topic: "Algorithms",
      note: "Stanford. Automata, languages, complexity — the theoretical backbone."
    }
  ],

  intermediate: [
    /* ── Machine Learning ─────────────────────────────────── */
    {
      title: "An Introduction to Statistical Learning",
      author: "James, Witten, Hastie, Tibshirani",
      url: "https://www.statlearning.com",
      format: "PDF",
      topic: "Machine Learning",
      note: "The applied ML textbook. Python edition available. Start here before Phase 02."
    },
    {
      title: "The Elements of Statistical Learning",
      author: "Hastie, Tibshirani, Friedman",
      url: "https://web.stanford.edu/~hastie/ElemStatLearn/",
      format: "PDF",
      topic: "Machine Learning",
      note: "Stanford. The rigorous version of ISLR. Reference for the math behind the methods."
    },
    {
      title: "Understanding Machine Learning: From Theory to Algorithms",
      author: "Shalev-Shwartz, Ben-David",
      url: "https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning",
      format: "PDF",
      topic: "Machine Learning",
      note: "Theory-first. If you want to understand WHY algorithms work, not just how to use them."
    },
    {
      title: "Machine Learning from Scratch",
      author: "Danny Friedman",
      url: "https://dafriedman97.github.io/mlbook/",
      format: "HTML, PDF, Jupyter",
      topic: "Machine Learning",
      note: "Builds every algorithm from math, then code. Mirrors the curriculum's Build It approach."
    },
    {
      title: "Interpretable Machine Learning",
      author: "Christoph Molnar",
      url: "https://christophm.github.io/interpretable-ml-book/",
      format: "HTML",
      topic: "Machine Learning",
      note: "SHAP, LIME, feature importance, model inspection. Essential for production ML."
    },
    {
      title: "Patterns, Predictions, and Actions",
      author: "Hardt, Recht",
      url: "https://mlstory.org/pdf/patterns.pdf",
      format: "PDF",
      topic: "Machine Learning",
      note: "ML through the lens of decision-making and causality. A different, important angle."
    },
    /* ── Deep Learning ────────────────────────────────────── */
    {
      title: "Deep Learning",
      author: "Goodfellow, Bengio, Courville",
      url: "https://www.deeplearningbook.org",
      format: "HTML",
      topic: "Deep Learning",
      note: "The canonical deep learning textbook. Read alongside Phases 03-07."
    },
    {
      title: "Dive into Deep Learning",
      author: "Zhang, Lipton, Li, Smola",
      url: "https://d2l.ai",
      format: "HTML, Jupyter",
      topic: "Deep Learning",
      note: "Interactive, multi-framework, updated constantly. The practical counterpart to Goodfellow."
    },
    {
      title: "Neural Networks and Deep Learning",
      author: "Michael Nielsen",
      url: "http://neuralnetworksanddeeplearning.com",
      format: "HTML",
      topic: "Deep Learning",
      note: "Best intro to neural networks that exists. Builds intuition from backprop up."
    },
    {
      title: "The Little Book of Deep Learning",
      author: "Francois Fleuret",
      url: "https://fleuret.org/public/lbdl.pdf",
      format: "PDF",
      topic: "Deep Learning",
      note: "Dense, precise, free. EPFL. 200 pages that cover everything essential."
    },
    {
      title: "Deep Learning for Coders with fastai and PyTorch",
      author: "Jeremy Howard, Sylvain Gugger",
      url: "https://github.com/fastai/fastbook",
      format: "Jupyter",
      topic: "Deep Learning",
      note: "Top-down, application-first. Best way to build things fast while learning the theory."
    },
    /* ── Computer Vision ──────────────────────────────────── */
    {
      title: "Computer Vision: Algorithms and Applications",
      author: "Richard Szeliski",
      url: "https://szeliski.org/Book/",
      format: "PDF, HTML",
      topic: "Computer Vision",
      note: "The definitive CV reference. Read alongside Phase 04."
    },
    {
      title: "Programming Computer Vision with Python",
      author: "Jan Erik Solem",
      url: "http://programmingcomputervision.com",
      format: "PDF",
      topic: "Computer Vision",
      note: "Hands-on CV with Python. Feature detection, segmentation, reconstruction."
    },
    /* ── NLP ──────────────────────────────────────────────── */
    {
      title: "Speech and Language Processing (3rd ed. draft)",
      author: "Jurafsky, Martin",
      url: "https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf",
      format: "PDF",
      topic: "NLP",
      note: "Stanford. The NLP bible. Covers transformers, LLMs, parsing, semantics. Read Phase 05-07."
    },
    {
      title: "Natural Language Processing with Python",
      author: "Bird, Klein, Loper",
      url: "http://www.nltk.org/book/",
      format: "HTML",
      topic: "NLP",
      note: "NLTK book. Classic NLP foundations with Python. Good companion to Phase 05."
    },
    /* ── Reinforcement Learning ───────────────────────────── */
    {
      title: "Reinforcement Learning: An Introduction",
      author: "Sutton, Barto",
      url: "http://incompleteideas.net/book/RLbook2020.pdf",
      format: "PDF",
      topic: "Reinforcement Learning",
      note: "The RL textbook. No substitute. Read alongside Phase 09."
    },
    {
      title: "Algorithms for Reinforcement Learning",
      author: "Csaba Szepesvari",
      url: "https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf",
      format: "PDF",
      topic: "Reinforcement Learning",
      note: "Concise, mathematical treatment of RL algorithms. Companion to Sutton & Barto."
    },
    /* ── Data Science ─────────────────────────────────────── */
    {
      title: "Python for Data Analysis",
      author: "Wes McKinney",
      url: "https://wesmckinney.com/book/",
      format: "HTML",
      topic: "Data Science",
      note: "Written by the creator of Pandas. The authoritative reference."
    },
    {
      title: "Mining of Massive Datasets",
      author: "Leskovec, Rajaraman, Ullman",
      url: "http://infolab.stanford.edu/~ullman/mmds/book.pdf",
      format: "PDF",
      topic: "Data Science",
      note: "Stanford. MapReduce, similarity, clustering, recommendations at scale."
    },
    {
      title: "Data Science at the Command Line",
      author: "Jeroen Janssens",
      url: "https://datascienceatthecommandline.com/2e/",
      format: "HTML",
      topic: "Data Science",
      note: "Shell tools for data pipelines. Underrated skill for any ML engineer."
    },
    /* ── Julia ────────────────────────────────────────────── */
    {
      title: "Julia Data Science",
      author: "Storopoli, Huijzer, Alonso",
      url: "https://juliadatascience.io",
      format: "HTML",
      topic: "Julia",
      note: "Data science in Julia from scratch. Pairs with the Julia lessons in Phases 01-02."
    },
    {
      title: "Think Julia",
      author: "Ben Lauwens, Allen Downey",
      url: "https://benlauwens.github.io/ThinkJulia.jl/latest/book.html",
      format: "HTML",
      topic: "Julia",
      note: "Best Julia intro. Clear, practical, builds real intuition."
    },
    {
      title: "Quantitative Economics with Julia",
      author: "Perla, Sargent, Stachurski",
      url: "https://julia.quantecon.org",
      format: "HTML",
      topic: "Julia",
      note: "Scientific computing with Julia. Numerical methods, optimization, dynamics."
    }
  ],

  advanced: [
    /* ── LLMs & Generative AI ─────────────────────────────── */
    {
      title: "How Transformer LLMs Work",
      author: "Jay Alammar, Maarten Grootendorst",
      url: "https://www.deeplearning.ai/short-courses/how-transformer-llms-work/",
      format: "Course (HTML)",
      topic: "LLMs",
      note: "DeepLearning.AI. The clearest explanation of transformer internals. Watch before Phase 07."
    },
    {
      title: "Hugging Face LLM Course",
      author: "Hugging Face",
      url: "https://huggingface.co/learn/llm-course/en/chapter1/1",
      format: "HTML",
      topic: "LLMs",
      note: "End-to-end LLM engineering: pretraining, fine-tuning, RLHF, deployment. Current."
    },
    {
      title: "Probabilistic Machine Learning: Advanced Topics",
      author: "Kevin Murphy",
      url: "https://probml.github.io/pml-book/book2.html",
      format: "PDF",
      topic: "LLMs",
      note: "State space models, diffusion, flows, transformers — the mathematical frontier."
    },
    {
      title: "Graph Representational Learning",
      author: "William Hamilton",
      url: "https://www.cs.mcgill.ca/~wlh/grl_book/",
      format: "PDF",
      topic: "LLMs",
      note: "McGill. GNNs, knowledge graphs, graph transformers. Increasingly central to LLMs."
    },
    {
      title: "Stanford CS224N: NLP with Deep Learning",
      author: "Christopher Manning",
      url: "https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ",
      format: "Video (YouTube)",
      topic: "LLMs",
      note: "Stanford. The gold standard NLP lecture series. Transformers, BERT, GPT from first principles."
    },
    /* ── AI Agents & Systems ──────────────────────────────── */
    {
      title: "How to Build Optimal AI Agents",
      author: "Tiago Monteiro",
      url: "https://www.freecodecamp.org/news/how-to-build-optimal-ai-agents-that-actually-work-a-handbook-for-devs/",
      format: "HTML",
      topic: "Agents",
      note: "Practical handbook for production agent systems. Pairs with Phases 13-16."
    },
    {
      title: "Artificial Intelligence: Foundations of Computational Agents (2nd ed.)",
      author: "Poole, Mackworth",
      url: "https://artint.info",
      format: "HTML",
      topic: "Agents",
      note: "Cambridge. Comprehensive agent theory — planning, search, learning, reasoning."
    },
    {
      title: "Introduction to Autonomous Robots",
      author: "Nikolaus Correll",
      url: "https://github.com/correll/Introduction-to-Autonomous-Robots/releases",
      format: "PDF",
      topic: "Agents",
      note: "Kinematics, perception, planning, control. The robotics foundation for autonomous agents."
    },
    /* ── MLOps & Production ───────────────────────────────── */
    {
      title: "Practitioners Guide to MLOps",
      author: "Google Cloud",
      url: "https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf",
      format: "PDF",
      topic: "MLOps",
      note: "Google. The production ML lifecycle: data, training, evaluation, deployment, monitoring."
    },
    {
      title: "Introduction to Machine Learning Systems",
      author: "Vijay Janapa Reddi",
      url: "https://www.mlsysbook.ai/assets/downloads/Machine-Learning-Systems.pdf",
      format: "PDF",
      topic: "MLOps",
      note: "Hardware-aware ML: efficiency, deployment, edge, quantization. Phase 17 prerequisite."
    },
    {
      title: "High Performance Computing Training",
      author: "LLNL",
      url: "https://hpc.llnl.gov/documentation/tutorials",
      format: "HTML",
      topic: "MLOps",
      note: "Lawrence Livermore National Lab. Parallel computing, MPI, OpenMP — for large-scale training."
    },
    {
      title: "Is Parallel Programming Hard?",
      author: "Paul McKenney",
      url: "https://www.kernel.org/pub/linux/kernel/people/paulmck/perfbook/perfbook.html",
      format: "HTML, PDF",
      topic: "MLOps",
      note: "The definitive guide to concurrent systems. Memory models, locking, RCU — for serious infra."
    },
    /* ── Rust ─────────────────────────────────────────────── */
    {
      title: "The Rust Programming Language",
      author: "Klabnik, Nichols, et al.",
      url: "https://doc.rust-lang.org/book",
      format: "HTML",
      topic: "Rust",
      note: "The official Rust book. Best language documentation that exists. Start here."
    },
    {
      title: "Rust by Example",
      author: "Rust community",
      url: "https://doc.rust-lang.org/stable/rust-by-example",
      format: "HTML",
      topic: "Rust",
      note: "Learn by reading and running examples. Companion to the Book."
    },
    {
      title: "Rust Atomics and Locks",
      author: "Mara Bos",
      url: "https://marabos.nl/atomics",
      format: "HTML",
      topic: "Rust",
      note: "Concurrency in Rust: atomics, mutexes, channels, memory ordering. For systems work."
    },
    {
      title: "Effective Rust",
      author: "David Drysdale",
      url: "https://www.lurklurk.org/effective-rust",
      format: "HTML, PDF",
      topic: "Rust",
      note: "35 specific ways to write better Rust. Bridges beginner and expert usage."
    },
    {
      title: "High Assurance Rust",
      author: "Tiemoko Ballo",
      url: "https://highassurance.rs",
      format: "HTML",
      topic: "Rust",
      note: "Rust for safety-critical systems. Embedded, verification, formal methods."
    },
    /* ── Prompt Engineering ───────────────────────────────── */
    {
      title: "Prompt Engineering Guide",
      author: "DAIR.AI",
      url: "https://www.promptingguide.ai",
      format: "HTML",
      topic: "Prompt Engineering",
      note: "Comprehensive, maintained, free. Chain-of-thought, RAG, agents, safety."
    },
    {
      title: "Prompt Engineering Guide",
      author: "LearnPrompting",
      url: "https://learnprompting.org/docs/introduction",
      format: "HTML",
      topic: "Prompt Engineering",
      note: "Broader coverage than DAIR.AI, more accessible. Good second reference."
    },
    /* ── TypeScript ───────────────────────────────────────── */
    {
      title: "TypeScript Deep Dive",
      author: "Basarat Ali Syed",
      url: "https://basarat.gitbooks.io/typescript/",
      format: "HTML",
      topic: "TypeScript",
      note: "The most complete TypeScript reference. Battle-tested by thousands."
    },
    {
      title: "Total TypeScript Essentials",
      author: "Matt Pocock",
      url: "https://www.totaltypescript.com/books/total-typescript-essentials",
      format: "HTML",
      topic: "TypeScript",
      note: "Modern TypeScript patterns: generics, type transformations, inference. Current."
    },
    {
      title: "TypeScript Handbook",
      author: "Microsoft",
      url: "https://www.typescriptlang.org/docs/handbook/intro.html",
      format: "HTML",
      topic: "TypeScript",
      note: "The official reference. Dry but authoritative. Keep as a lookup."
    },
    /* ── Professional Development ─────────────────────────── */
    {
      title: "Software Engineering at Google",
      author: "Winters, Manshreck, Wright",
      url: "https://abseil.io/resources/swe-book",
      format: "HTML",
      topic: "Professional",
      note: "How Google builds software at scale. Testing, reviews, deprecation, culture."
    },
    {
      title: "A Selective Overview of Deep Learning",
      author: "Fan, Ma, Zhong",
      url: "https://arxiv.org/abs/1904.05526",
      format: "PDF",
      topic: "Research",
      note: "Princeton. Statistical theory behind deep learning. For understanding convergence and generalization."
    }
  ]
};
