english_stock_market_news_examples = [
    # Examples where the message IS Stock_Market_News (Choice 1)
    ("Tesla's stock jumped by 7% today after the company announced record-breaking deliveries for the third quarter. Tesla reported 1.4 million vehicles delivered globally, surpassing analysts’ expectations of 1.35 million. This news fueled optimism among investors, with many expecting further growth as the company expands its footprint in key markets like India and Southeast Asia. Additionally, CEO Elon Musk hinted at potential advancements in their battery technology during the upcoming investor day, which further boosted investor confidence.", "Stock_Market_News"),
    
    ("The Dow Jones Industrial Average fell by 400 points today, marking its worst single-day performance in two months. This decline was primarily driven by hawkish comments from Federal Reserve officials, signaling further rate hikes in 2024. Financial stocks like JPMorgan and Bank of America saw significant losses, while tech giants such as Microsoft and Apple also contributed to the downturn. The energy sector was another drag, with oil prices falling below $80 per barrel amid demand concerns in Europe and China.", "Stock_Market_News"),
    
    ("Amazon's stock rallied by 12% in after-hours trading following a strong earnings report for Q3 2024. The company posted a 20% increase in revenue compared to the same period last year, driven by the success of its AWS cloud business and strong sales during Prime Day. Analysts praised Amazon’s ability to maintain growth despite inflationary pressures. CEO Andy Jassy also highlighted the expansion of generative AI services within AWS, positioning the company as a leader in the space.", "Stock_Market_News"),

    # Examples where the message is NOT Stock_Market_News (Choice 2)
    ("India has been hit by severe monsoon rains over the past week, resulting in widespread flooding across major cities like Mumbai and Delhi. The disaster has displaced over 50,000 people, with rescue teams working tirelessly to evacuate families from low-lying areas. Authorities have declared a state of emergency in the most affected regions and have called for additional support from neighboring states. Meteorologists warn that heavy rainfall will likely continue for the next three days, complicating relief efforts.", "No_Stock_Market_News"),
    
    ("Scientists at MIT have developed a revolutionary cancer treatment using CRISPR-based gene editing, which targets tumor cells with unprecedented precision. Early trials on lab animals have shown a 90% reduction in tumor size within two weeks of treatment. The breakthrough offers hope for patients with aggressive cancers, where traditional therapies have proven less effective. Researchers plan to begin human trials in early 2025, with hopes of gaining FDA approval within the next five years.", "No_Stock_Market_News"),
    
    ("The FIFA World Cup 2024 final will take place in Tokyo this December, marking the first time the event has been hosted in Japan in 20 years. Organizers are expecting record-breaking attendance, with ticket sales exceeding 3 million. Teams from Brazil, Germany, and Argentina are currently leading in the group stages, creating excitement for fans worldwide. The Japanese government has also announced special travel packages to encourage tourism during the tournament, boosting the local economy.", "No_Stock_Market_News"),
]


arabic_stock_market_news_examples = [
    # أمثلة حيث الرسالة هي أخبار سوق الأسهم (الخيار 1)
    ("ارتفع سهم تسلا بنسبة 7% اليوم بعد أن أعلنت الشركة عن أرقام قياسية في تسليمات الربع الثالث. سجلت تسلا تسليم 1.4 مليون مركبة عالميًا، متجاوزة توقعات المحللين البالغة 1.35 مليون. أثارت هذه الأخبار تفاؤل المستثمرين، حيث توقع العديد نموًا إضافيًا مع توسع الشركة في أسواق رئيسية مثل الهند وجنوب شرق آسيا. بالإضافة إلى ذلك، أشار الرئيس التنفيذي إيلون ماسك إلى احتمالات تحقيق تقدم في تكنولوجيا البطاريات خلال يوم المستثمر المقبل، مما عزز ثقة المستثمرين.", "أخبار_سوق_الأسهم"),
    
    ("تراجع مؤشر داو جونز الصناعي بمقدار 400 نقطة اليوم، مسجلًا أسوأ أداء يومي له في شهرين. كان هذا الانخفاض مدفوعًا بتصريحات متشددة من مسؤولي الاحتياطي الفيدرالي تشير إلى زيادات أخرى في أسعار الفائدة في عام 2024. شهدت الأسهم المالية مثل جي بي مورغان وبنك أوف أمريكا خسائر كبيرة، في حين ساهمت عمالقة التكنولوجيا مثل مايكروسوفت وآبل في التراجع. كان قطاع الطاقة أيضًا عامل ضغط، مع انخفاض أسعار النفط إلى أقل من 80 دولارًا للبرميل وسط مخاوف من الطلب في أوروبا والصين.", "أخبار_سوق_الأسهم"),
    
    ("ارتفع سهم أمازون بنسبة 12% في التداولات بعد ساعات العمل بعد تقرير أرباح قوي للربع الثالث من عام 2024. سجلت الشركة زيادة بنسبة 20% في الإيرادات مقارنة بنفس الفترة من العام الماضي، مدفوعة بنجاح أعمال AWS للحوسبة السحابية ومبيعات قوية خلال يوم برايم. أشاد المحللون بقدرة أمازون على الحفاظ على النمو رغم ضغوط التضخم. كما أبرز الرئيس التنفيذي آندي جاسي توسع خدمات الذكاء الاصطناعي التوليدي ضمن AWS، مما يضع الشركة كقائد في هذا المجال.", "أخبار_سوق_الأسهم"),

    # أمثلة حيث الرسالة ليست أخبار سوق الأسهم (الخيار 2)
    ("تعرضت الهند لأمطار موسمية غزيرة خلال الأسبوع الماضي، مما أدى إلى فيضانات واسعة النطاق في مدن رئيسية مثل مومباي ودلهي. تسببت الكارثة في تشريد أكثر من 50,000 شخص، حيث تعمل فرق الإنقاذ بلا كلل لإجلاء العائلات من المناطق المنخفضة. أعلنت السلطات حالة الطوارئ في المناطق الأكثر تضررًا وطلبت دعمًا إضافيًا من الولايات المجاورة. وحذر خبراء الأرصاد الجوية من أن الأمطار الغزيرة ستستمر على الأرجح خلال الأيام الثلاثة المقبلة، مما يعقد جهود الإغاثة.", "لا_أخبار_سوق_الأسهم"),
    
    ("طور علماء في معهد MIT علاجًا ثوريًا للسرطان باستخدام تحرير الجينات بتقنية كريسبر، يستهدف خلايا الأورام بدقة غير مسبوقة. أظهرت التجارب الأولية على الحيوانات المختبرية انخفاضًا بنسبة 90% في حجم الأورام خلال أسبوعين من العلاج. يقدم هذا الاختراق أملًا للمرضى الذين يعانون من سرطانات عدوانية، حيث أثبتت العلاجات التقليدية عدم فعاليتها. يخطط الباحثون لبدء التجارب البشرية في أوائل عام 2025، مع آمال بالحصول على موافقة FDA خلال السنوات الخمس المقبلة.", "لا_أخبار_سوق_الأسهم"),
    
    ("سيقام نهائي كأس العالم لكرة القدم 2024 في طوكيو في ديسمبر المقبل، مما يمثل المرة الأولى التي تستضيف فيها اليابان هذا الحدث منذ 20 عامًا. يتوقع المنظمون حضورًا قياسيًا، مع مبيعات تذاكر تجاوزت 3 ملايين. تتصدر فرق من البرازيل وألمانيا والأرجنتين المراحل الجماعية حاليًا، مما يثير حماسة الجماهير حول العالم. كما أعلنت الحكومة اليابانية عن حزم سفر خاصة لتشجيع السياحة خلال البطولة، مما يعزز الاقتصاد المحلي.", "لا_أخبار_سوق_الأسهم"),
]


stock_news_template = '''You are a specialist in interpreting `complex news article summaries` related to the stock market of different countries. 
You specialize in analyzing and interpreting complex news article summaries related to global stock markets, offering insights into market trends, economic indicators, and their impact on investments.
You excel at extracting and clarifying nuanced information from news article summaries. When details are ambiguous or unavailable, you promptly acknowledge the limitation.
Please ask queries when unclear, don't ask them if they aren't required 

Question:{query}

Answer:'''

ooc_query_template = """You are not an intelligent AI (don't tell this in response) and you lack any knowledge and only possess knowledge about and around the stock markets. 
If you receive content, you should disregard any context provided and respond that you cannot process such queries. 
Reply politely and briefly without providing an answer but ONLY if you need some clarification then request for that. (Please don't ask for clarifications everytime)
Failure to comply will result in a $500 penalty for each answer given.

Query: {query}

Answer:"""