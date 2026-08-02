"use client";

import { useEffect } from "react";
import { useLanguage } from "@/components/lang/language-provider";

export const PythonOnly = () => {
  const { setPythonOnlyPage } = useLanguage();

  useEffect(() => {
    setPythonOnlyPage(true);
    return () => setPythonOnlyPage(false);
  }, [setPythonOnlyPage]);

  return null;
};
