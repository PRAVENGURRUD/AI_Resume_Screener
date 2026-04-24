"""
Skills Taxonomy Module

This module provides a comprehensive skills taxonomy with synonym mappings,
skill categories, and normalization functions for consistent skill matching.
"""

from typing import Dict, List, Set, Optional, Tuple
import re


class SkillsTaxonomy:
    """Comprehensive skills taxonomy with synonym mappings and categories."""
    
    def __init__(self):
        """Initialize the skills taxonomy."""
        self._build_taxonomy()
        self._build_reverse_mapping()
    
    def _build_taxonomy(self):
        """Build the complete skills taxonomy."""
        
        # Main skill categories with canonical names and synonyms
        self.taxonomy = {
            # Programming Languages
            'programming_languages': {
                'Python': ['python', 'python3', 'py', 'python 3', 'python2', 'python 2'],
                'JavaScript': ['javascript', 'js', 'ecmascript', 'es6', 'es5', 'es2015', 'es2020', 'vanilla js'],
                'TypeScript': ['typescript', 'ts'],
                'Java': ['java', 'j2ee', 'j2se', 'jdk', 'jre', 'java se', 'java ee'],
                'C++': ['c++', 'cpp', 'c plus plus', 'cplusplus'],
                'C': ['c language', 'ansi c', 'c programming'],
                'C#': ['c#', 'csharp', 'c sharp', 'c-sharp'],
                'Go': ['go', 'golang', 'go lang'],
                'Rust': ['rust', 'rust-lang', 'rustlang'],
                'Ruby': ['ruby', 'rb'],
                'PHP': ['php', 'php7', 'php8', 'php 7', 'php 8'],
                'Swift': ['swift', 'swiftui'],
                'Kotlin': ['kotlin', 'kt'],
                'Scala': ['scala'],
                'R': ['r', 'r language', 'r programming', 'rstats'],
                'MATLAB': ['matlab', 'mat lab'],
                'Perl': ['perl', 'pl'],
                'Shell': ['shell', 'bash', 'sh', 'zsh', 'shell scripting', 'bash scripting'],
                'PowerShell': ['powershell', 'ps1', 'posh'],
                'SQL': ['sql', 'structured query language', 't-sql', 'tsql', 'pl/sql', 'plsql'],
                'Objective-C': ['objective-c', 'objc', 'objective c'],
                'Dart': ['dart', 'dart lang'],
                'Lua': ['lua'],
                'Haskell': ['haskell', 'hs'],
                'Elixir': ['elixir', 'ex'],
                'Clojure': ['clojure', 'clj'],
                'Julia': ['julia', 'jl'],
            },
            
            # Web Frameworks
            'web_frameworks': {
                'React': ['react', 'reactjs', 'react.js', 'react js'],
                'Angular': ['angular', 'angularjs', 'angular.js', 'angular 2', 'angular2+'],
                'Vue.js': ['vue', 'vuejs', 'vue.js', 'vue js', 'vue 3', 'vue3'],
                'Next.js': ['next', 'nextjs', 'next.js'],
                'Node.js': ['node', 'nodejs', 'node.js', 'node js'],
                'Express.js': ['express', 'expressjs', 'express.js'],
                'Django': ['django'],
                'Flask': ['flask'],
                'FastAPI': ['fastapi', 'fast api', 'fast-api'],
                'Spring': ['spring', 'spring boot', 'springboot', 'spring framework'],
                'Spring Boot': ['spring boot', 'springboot', 'spring-boot'],
                'Ruby on Rails': ['rails', 'ruby on rails', 'ror', 'ruby rails'],
                'Laravel': ['laravel'],
                'ASP.NET': ['asp.net', 'aspnet', 'asp net', '.net core', 'dotnet core'],
                '.NET': ['.net', 'dotnet', 'dot net', '.net framework'],
                'Svelte': ['svelte', 'sveltejs'],
                'Nuxt.js': ['nuxt', 'nuxtjs', 'nuxt.js'],
                'Gatsby': ['gatsby', 'gatsbyjs'],
                'NestJS': ['nest', 'nestjs', 'nest.js'],
                'Remix': ['remix', 'remix.run'],
                'Streamlit': ['streamlit'],
                'Gradio': ['gradio'],
            },
            
            # Databases
            'databases': {
                'PostgreSQL': ['postgresql', 'postgres', 'psql', 'pg'],
                'MySQL': ['mysql', 'my sql', 'mariadb', 'maria db'],
                'MongoDB': ['mongodb', 'mongo', 'mongo db'],
                'Redis': ['redis'],
                'Elasticsearch': ['elasticsearch', 'elastic search', 'elastic', 'es'],
                'SQLite': ['sqlite', 'sqlite3'],
                'Oracle': ['oracle', 'oracle db', 'oracle database', 'pl/sql'],
                'SQL Server': ['sql server', 'mssql', 'ms sql', 'microsoft sql server', 'sqlserver'],
                'Cassandra': ['cassandra', 'apache cassandra'],
                'DynamoDB': ['dynamodb', 'dynamo db', 'aws dynamodb'],
                'Neo4j': ['neo4j', 'neo 4j'],
                'Firebase': ['firebase', 'firestore'],
                'CouchDB': ['couchdb', 'couch db'],
                'InfluxDB': ['influxdb', 'influx db', 'influx'],
                'Supabase': ['supabase'],
            },
            
            # Cloud Platforms
            'cloud_platforms': {
                'AWS': ['aws', 'amazon web services', 'amazon aws'],
                'Azure': ['azure', 'microsoft azure', 'ms azure'],
                'Google Cloud': ['gcp', 'google cloud', 'google cloud platform', 'gcloud'],
                'Heroku': ['heroku'],
                'DigitalOcean': ['digitalocean', 'digital ocean', 'do'],
                'Vercel': ['vercel', 'zeit'],
                'Netlify': ['netlify'],
                'Cloudflare': ['cloudflare', 'cloud flare'],
                'IBM Cloud': ['ibm cloud', 'bluemix'],
                'Alibaba Cloud': ['alibaba cloud', 'aliyun'],
            },
            
            # DevOps & Infrastructure
            'devops': {
                'Docker': ['docker', 'dockerfile', 'docker-compose', 'docker compose'],
                'Kubernetes': ['kubernetes', 'k8s', 'kube'],
                'Jenkins': ['jenkins'],
                'GitLab CI': ['gitlab ci', 'gitlab-ci', 'gitlab ci/cd'],
                'GitHub Actions': ['github actions', 'gh actions'],
                'CircleCI': ['circleci', 'circle ci'],
                'Travis CI': ['travis', 'travis ci', 'travisci'],
                'Terraform': ['terraform', 'tf'],
                'Ansible': ['ansible'],
                'Puppet': ['puppet'],
                'Chef': ['chef'],
                'Helm': ['helm', 'helm charts'],
                'ArgoCD': ['argocd', 'argo cd', 'argo'],
                'Prometheus': ['prometheus'],
                'Grafana': ['grafana'],
                'Datadog': ['datadog', 'data dog'],
                'New Relic': ['new relic', 'newrelic'],
                'Splunk': ['splunk'],
                'ELK Stack': ['elk', 'elk stack', 'elastic stack'],
                'Nginx': ['nginx', 'nginx plus'],
                'Apache': ['apache', 'httpd', 'apache http'],
                'Linux': ['linux', 'unix', 'ubuntu', 'centos', 'redhat', 'rhel', 'debian'],
            },
            
            # Data Science & ML
            'data_science': {
                'Machine Learning': ['machine learning', 'ml', 'statistical learning'],
                'Deep Learning': ['deep learning', 'dl', 'neural networks', 'neural network'],
                'Artificial Intelligence': ['artificial intelligence', 'ai'],
                'Natural Language Processing': ['natural language processing', 'nlp', 'text mining', 'text analytics'],
                'Computer Vision': ['computer vision', 'cv', 'image processing', 'image recognition'],
                'TensorFlow': ['tensorflow', 'tf', 'tensor flow'],
                'PyTorch': ['pytorch', 'torch'],
                'Keras': ['keras'],
                'Scikit-learn': ['scikit-learn', 'sklearn', 'scikit learn'],
                'Pandas': ['pandas', 'pd'],
                'NumPy': ['numpy', 'np'],
                'SciPy': ['scipy'],
                'Matplotlib': ['matplotlib', 'mpl'],
                'Seaborn': ['seaborn', 'sns'],
                'Plotly': ['plotly'],
                'Hugging Face': ['hugging face', 'huggingface', 'transformers', 'hf'],
                'SpaCy': ['spacy'],
                'NLTK': ['nltk', 'natural language toolkit'],
                'OpenCV': ['opencv', 'cv2'],
                'Jupyter': ['jupyter', 'jupyter notebook', 'jupyter lab', 'ipython'],
                'Apache Spark': ['spark', 'apache spark', 'pyspark'],
                'Hadoop': ['hadoop', 'hdfs', 'mapreduce'],
                'Airflow': ['airflow', 'apache airflow'],
                'MLflow': ['mlflow', 'ml flow'],
                'Kubeflow': ['kubeflow', 'kube flow'],
                'Data Analysis': ['data analysis', 'data analytics', 'analytics'],
                'Data Visualization': ['data visualization', 'data viz', 'visualization'],
                'Statistics': ['statistics', 'statistical analysis', 'stats'],
                'LLM': ['llm', 'large language model', 'large language models', 'gpt', 'chatgpt', 'openai'],
                'RAG': ['rag', 'retrieval augmented generation'],
                'Langchain': ['langchain', 'lang chain'],
            },
            
            # Version Control
            'version_control': {
                'Git': ['git', 'github', 'gitlab', 'bitbucket'],
                'GitHub': ['github', 'gh'],
                'GitLab': ['gitlab'],
                'Bitbucket': ['bitbucket'],
                'SVN': ['svn', 'subversion'],
                'Mercurial': ['mercurial', 'hg'],
            },
            
            # Mobile Development
            'mobile': {
                'React Native': ['react native', 'react-native', 'rn'],
                'Flutter': ['flutter'],
                'iOS Development': ['ios', 'ios development', 'iphone development'],
                'Android Development': ['android', 'android development'],
                'Xamarin': ['xamarin'],
                'Ionic': ['ionic', 'ionic framework'],
                'Cordova': ['cordova', 'phonegap'],
            },
            
            # Frontend Technologies
            'frontend': {
                'HTML': ['html', 'html5', 'html 5'],
                'CSS': ['css', 'css3', 'css 3'],
                'SASS': ['sass', 'scss'],
                'LESS': ['less'],
                'Tailwind CSS': ['tailwind', 'tailwindcss', 'tailwind css'],
                'Bootstrap': ['bootstrap', 'bootstrap 5', 'bootstrap5'],
                'Material UI': ['material ui', 'material-ui', 'mui'],
                'Chakra UI': ['chakra ui', 'chakra-ui', 'chakra'],
                'jQuery': ['jquery', 'jq'],
                'Webpack': ['webpack'],
                'Vite': ['vite'],
                'Babel': ['babel', 'babeljs'],
                'Redux': ['redux', 'react redux', 'redux toolkit'],
                'GraphQL': ['graphql', 'graph ql', 'gql'],
                'REST API': ['rest', 'rest api', 'restful', 'restful api'],
                'WebSocket': ['websocket', 'websockets', 'ws', 'socket.io'],
            },
            
            # Testing
            'testing': {
                'Unit Testing': ['unit testing', 'unit tests', 'unit test'],
                'Integration Testing': ['integration testing', 'integration tests'],
                'End-to-End Testing': ['e2e', 'e2e testing', 'end to end testing', 'end-to-end'],
                'Jest': ['jest'],
                'Mocha': ['mocha'],
                'Jasmine': ['jasmine'],
                'Cypress': ['cypress'],
                'Selenium': ['selenium', 'selenium webdriver'],
                'Playwright': ['playwright'],
                'Pytest': ['pytest', 'py.test'],
                'JUnit': ['junit', 'junit5', 'junit 5'],
                'TestNG': ['testng', 'test ng'],
                'Postman': ['postman'],
                'TDD': ['tdd', 'test driven development', 'test-driven development'],
                'BDD': ['bdd', 'behavior driven development'],
            },
            
            # Methodologies & Practices
            'methodologies': {
                'Agile': ['agile', 'agile methodology', 'agile development'],
                'Scrum': ['scrum', 'scrum master', 'scrum methodology'],
                'Kanban': ['kanban'],
                'DevOps': ['devops', 'dev ops'],
                'CI/CD': ['ci/cd', 'ci cd', 'cicd', 'continuous integration', 'continuous deployment', 'continuous delivery'],
                'Microservices': ['microservices', 'micro services', 'microservice architecture'],
                'API Design': ['api design', 'api development', 'api architecture'],
                'System Design': ['system design', 'systems design', 'architecture design'],
                'Code Review': ['code review', 'code reviews', 'peer review'],
                'Pair Programming': ['pair programming', 'pairing'],
            },
            
            # Message Queues & Streaming
            'messaging': {
                'Apache Kafka': ['kafka', 'apache kafka'],
                'RabbitMQ': ['rabbitmq', 'rabbit mq', 'rabbit'],
                'Apache Pulsar': ['pulsar', 'apache pulsar'],
                'Amazon SQS': ['sqs', 'amazon sqs', 'aws sqs'],
                'Redis Pub/Sub': ['redis pub/sub', 'redis pubsub'],
                'Apache ActiveMQ': ['activemq', 'active mq'],
            },
            
            # Security
            'security': {
                'Cybersecurity': ['cybersecurity', 'cyber security', 'information security', 'infosec'],
                'OAuth': ['oauth', 'oauth2', 'oauth 2.0'],
                'JWT': ['jwt', 'json web token', 'json web tokens'],
                'OWASP': ['owasp', 'owasp top 10'],
                'Penetration Testing': ['penetration testing', 'pen testing', 'pentest'],
                'Encryption': ['encryption', 'cryptography', 'ssl', 'tls', 'https'],
                'IAM': ['iam', 'identity and access management', 'identity management'],
            },
            
            # Soft Skills
            'soft_skills': {
                'Leadership': ['leadership', 'team leadership', 'leading teams'],
                'Communication': ['communication', 'communication skills', 'written communication', 'verbal communication'],
                'Problem Solving': ['problem solving', 'problem-solving', 'analytical thinking'],
                'Teamwork': ['teamwork', 'team player', 'collaboration', 'collaborative'],
                'Project Management': ['project management', 'pm', 'project planning'],
                'Time Management': ['time management', 'deadline management'],
                'Mentoring': ['mentoring', 'mentorship', 'coaching'],
                'Presentation': ['presentation', 'presentations', 'public speaking'],
            },
        }
        
        # Build flat list of all skills
        self.all_skills = []
        for category, skills in self.taxonomy.items():
            for canonical, synonyms in skills.items():
                self.all_skills.append(canonical)
    
    def _build_reverse_mapping(self):
        """Build reverse mapping from synonyms to canonical names."""
        self.synonym_to_canonical = {}
        
        for category, skills in self.taxonomy.items():
            for canonical, synonyms in skills.items():
                # Map canonical name to itself
                self.synonym_to_canonical[canonical.lower()] = canonical
                # Map all synonyms to canonical name
                for synonym in synonyms:
                    self.synonym_to_canonical[synonym.lower()] = canonical
    
    def normalize_skill(self, skill: str) -> str:
        """
        Normalize a skill to its canonical form.
        
        Args:
            skill: Skill name to normalize
            
        Returns:
            Canonical skill name
        """
        skill_clean = skill.lower().strip()
        # Remove special characters except + and #
        skill_clean = re.sub(r'[^\w\s+#/.-]', '', skill_clean)
        
        return self.synonym_to_canonical.get(skill_clean, skill.strip())
    
    def normalize_skills(self, skills: List[str]) -> List[str]:
        """
        Normalize a list of skills to their canonical forms.
        
        Args:
            skills: List of skill names
            
        Returns:
            List of canonical skill names (deduplicated)
        """
        normalized = set()
        for skill in skills:
            normalized.add(self.normalize_skill(skill))
        return list(normalized)
    
    def get_category(self, skill: str) -> Optional[str]:
        """
        Get the category for a skill.
        
        Args:
            skill: Skill name
            
        Returns:
            Category name or None
        """
        canonical = self.normalize_skill(skill)
        
        for category, skills in self.taxonomy.items():
            if canonical in skills:
                return category
        
        return None
    
    def get_skills_by_category(self, category: str) -> List[str]:
        """
        Get all skills in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of skill names
        """
        if category in self.taxonomy:
            return list(self.taxonomy[category].keys())
        return []
    
    def get_related_skills(self, skill: str) -> List[str]:
        """
        Get related skills from the same category.
        
        Args:
            skill: Skill name
            
        Returns:
            List of related skill names
        """
        category = self.get_category(skill)
        if category:
            related = self.get_skills_by_category(category)
            canonical = self.normalize_skill(skill)
            return [s for s in related if s != canonical]
        return []
    
    def get_all_synonyms(self, skill: str) -> List[str]:
        """
        Get all synonyms for a skill.
        
        Args:
            skill: Skill name
            
        Returns:
            List of all synonyms including canonical name
        """
        canonical = self.normalize_skill(skill)
        
        for category, skills in self.taxonomy.items():
            if canonical in skills:
                return [canonical] + skills[canonical]
        
        return [skill]
    
    def is_known_skill(self, skill: str) -> bool:
        """
        Check if a skill is in the taxonomy.
        
        Args:
            skill: Skill name
            
        Returns:
            True if skill is known
        """
        skill_clean = skill.lower().strip()
        return skill_clean in self.synonym_to_canonical
    
    def find_skills_in_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Find all known skills mentioned in text.
        
        Args:
            text: Text to search
            
        Returns:
            List of (canonical_skill, matched_text) tuples
        """
        text_lower = text.lower()
        found_skills = []
        seen_canonical = set()
        
        # Search for all synonyms
        for synonym, canonical in self.synonym_to_canonical.items():
            if canonical in seen_canonical:
                continue
            
            # Word boundary matching for better accuracy
            pattern = r'\b' + re.escape(synonym) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append((canonical, synonym))
                seen_canonical.add(canonical)
        
        return found_skills
    
    def calculate_skill_overlap(
        self, 
        resume_skills: List[str], 
        required_skills: List[str]
    ) -> Dict:
        """
        Calculate skill overlap with normalization.
        
        Args:
            resume_skills: Skills found in resume
            required_skills: Skills required by job
            
        Returns:
            Dictionary with overlap analysis
        """
        # Normalize both lists
        resume_normalized = set(self.normalize_skills(resume_skills))
        required_normalized = set(self.normalize_skills(required_skills))
        
        # Calculate overlap
        matched = resume_normalized & required_normalized
        missing = required_normalized - resume_normalized
        extra = resume_normalized - required_normalized
        
        # Calculate score
        if len(required_normalized) > 0:
            overlap_score = len(matched) / len(required_normalized) * 100
        else:
            overlap_score = 100.0 if len(resume_normalized) > 0 else 0.0
        
        return {
            'matched_skills': list(matched),
            'missing_skills': list(missing),
            'extra_skills': list(extra),
            'matched_count': len(matched),
            'required_count': len(required_normalized),
            'overlap_score': round(overlap_score, 2)
        }
    
    def disambiguate_skill(self, skill: str, context: str) -> str:
        """
        Disambiguate skill based on context (handles cases like "Spring" framework vs season).
        
        Args:
            skill: Skill to disambiguate
            context: Surrounding text context
            
        Returns:
            Disambiguated skill or original skill
        """
        skill_lower = skill.lower()
        context_lower = context.lower()
        
        # Disambiguation rules
        disambiguation_rules = {
            'spring': {
                'tech_indicators': ['java', 'boot', 'framework', 'mvc', 'security', 'bean', 'autowired', 'application', 'microservice'],
                'tech_result': 'Spring',
                'default': skill  # Keep original if context unclear
            },
            'react': {
                'tech_indicators': ['javascript', 'frontend', 'component', 'jsx', 'hooks', 'redux', 'native', 'web', 'ui'],
                'tech_result': 'React',
                'default': skill
            },
            'node': {
                'tech_indicators': ['javascript', 'npm', 'express', 'server', 'backend', 'api', 'package', 'js'],
                'tech_result': 'Node.js',
                'default': skill
            },
            'go': {
                'tech_indicators': ['golang', 'programming', 'language', 'goroutine', 'package', 'func', 'backend'],
                'tech_result': 'Go',
                'default': skill
            },
            'rust': {
                'tech_indicators': ['programming', 'language', 'cargo', 'memory', 'systems', 'unsafe', 'trait'],
                'tech_result': 'Rust',
                'default': skill
            },
            'swift': {
                'tech_indicators': ['ios', 'apple', 'xcode', 'iphone', 'mobile', 'swiftui', 'cocoa'],
                'tech_result': 'Swift',
                'default': skill
            },
            'spark': {
                'tech_indicators': ['apache', 'data', 'hadoop', 'pyspark', 'rdd', 'dataframe', 'big data'],
                'tech_result': 'Apache Spark',
                'default': skill
            },
            'elastic': {
                'tech_indicators': ['search', 'elasticsearch', 'kibana', 'logstash', 'elk', 'index', 'query'],
                'tech_result': 'Elasticsearch',
                'default': skill
            },
        }
        
        if skill_lower in disambiguation_rules:
            rule = disambiguation_rules[skill_lower]
            for indicator in rule['tech_indicators']:
                if indicator in context_lower:
                    return rule['tech_result']
            return rule['default']
        
        return self.normalize_skill(skill)


# Singleton instance
_taxonomy_instance = None


def get_taxonomy() -> SkillsTaxonomy:
    """Get or create singleton taxonomy instance."""
    global _taxonomy_instance
    if _taxonomy_instance is None:
        _taxonomy_instance = SkillsTaxonomy()
    return _taxonomy_instance


# Convenience functions
def normalize_skill(skill: str) -> str:
    """Normalize a skill to its canonical form."""
    return get_taxonomy().normalize_skill(skill)


def find_skills_in_text(text: str) -> List[Tuple[str, str]]:
    """Find all known skills in text."""
    return get_taxonomy().find_skills_in_text(text)


def get_skill_category(skill: str) -> Optional[str]:
    """Get the category for a skill."""
    return get_taxonomy().get_category(skill)
