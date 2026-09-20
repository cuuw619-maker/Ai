use super::core::AppPage;

pub fn is_russian(language: &str) -> bool {
    language.eq_ignore_ascii_case("Russian") || language.eq_ignore_ascii_case("Русский")
}

pub fn text(language: &str, key: &str) -> &'static str {
    if is_russian(language) {
        match key {
            "HOME" => "ГЛАВНАЯ",
            "CHAT" => "ЧАТ",
            "TRAIN" => "ОБУЧЕНИЕ",
            "MODEL" => "МОДЕЛЬ",
            "DATASET" => "ДАТАСЕТ",
            "MEMORY" => "ПАМЯТЬ",
            "EVALUATION" => "ОЦЕНКА",
            "LOGS" => "ЛОГИ",
            "SETTINGS" => "НАСТРОЙКИ",
            "SYSTEM" => "СИСТЕМА",
            "WEB LEARNING" => "ВЕБ-ОБУЧЕНИЕ",
            "Own Neural Engine" => "Собственный нейронный движок",
            "LOCAL / FROM SCRATCH" => "ЛОКАЛЬНО / С НУЛЯ",
            "Language" => "Язык",
            "UI Language" => "Язык интерфейса",
            "AI/Data Languages" => "Языки ИИ/данных",
            "English" => "English",
            "Russian" => "Русский",
            "Self Test" => "Самотестирование",
            "SAVE CONFIG" => "СОХРАНИТЬ КОНФИГ",
            "Source URL" => "URL источника",
            "ADD SOURCE" => "ДОБАВИТЬ ИСТОЧНИК",
            "AUTONOMOUS LEARNING" => "АВТОНОМНОЕ ОБУЧЕНИЕ",
            "START" => "ЗАПУСТИТЬ",
            "PAUSE WEB" => "ПАУЗА ВЕБ",
            "STOP" => "СТОП",
            "SCAN NOW" => "СКАНИРОВАТЬ",
            "ROUTING ACTIVITY" => "АКТИВНОСТЬ МАРШРУТИЗАЦИИ",
            _ => "",
        }
    } else {
        match key {
            "HOME" | "CHAT" | "TRAIN" | "MODEL" | "DATASET" | "MEMORY" | "EVALUATION"
            | "LOGS" | "SETTINGS" | "SYSTEM" | "WEB LEARNING" => match key {
                "HOME" => "HOME",
                "CHAT" => "CHAT",
                "TRAIN" => "TRAIN",
                "MODEL" => "MODEL",
                "DATASET" => "DATASET",
                "MEMORY" => "MEMORY",
                "EVALUATION" => "EVALUATION",
                "LOGS" => "LOGS",
                "SETTINGS" => "SETTINGS",
                "SYSTEM" => "SYSTEM",
                "WEB LEARNING" => "WEB LEARNING",
                _ => "",
            },
            _ => "",
        }
    }
}

pub fn page_label(page: AppPage, language: &str) -> &'static str {
    text(language, page.name())
}

pub fn page_title(page: AppPage, language: &str) -> &'static str {
    if is_russian(language) {
        match page {
            AppPage::Home => "Главная",
            AppPage::Chat => "Чат",
            AppPage::Train => "Обучение",
            AppPage::Model => "Модель",
            AppPage::Dataset => "Датасет",
            AppPage::Memory => "Память",
            AppPage::Evaluation => "Оценка",
            AppPage::Logs => "Логи",
            AppPage::Settings => "Настройки",
            AppPage::System => "Система и диагностика",
            AppPage::WebLearning => "Веб-обучение",
        }
    } else {
        match page {
            AppPage::Home => "Home",
            AppPage::Chat => "Chat",
            AppPage::Train => "Training",
            AppPage::Model => "Model",
            AppPage::Dataset => "Dataset",
            AppPage::Memory => "Memory",
            AppPage::Evaluation => "Evaluation",
            AppPage::Logs => "Logs",
            AppPage::Settings => "Settings",
            AppPage::System => "System & Diagnostics",
            AppPage::WebLearning => "Web Learning",
        }
    }
}
